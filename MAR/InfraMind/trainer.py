"""
InfraMind Trainer — two-level training for hierarchical CMDP router.

PPO-Lagrangian formulation with λ warmup:
    maximize   E[quality]
    subject to E[cost] <= 1   (where cost = latency / budget)

Shared Lagrange multiplier λ adapts the quality-cost tradeoff automatically.
When the policy violates the budget, λ increases → both levels prioritize cost.
When the policy is within budget, λ decreases → both levels focus on quality.

λ warmup: λ is held at 0 for the first `warmup_epochs` epochs so the policy
first learns quality maximization (which models/strategies solve problems).
This prevents the bootstrap trap where λ rises before budget-conditional
behavior can develop, collapsing expensive-but-accurate models globally.

Planner training (REINFORCE + Lagrangian):
    quality_reward = 1.0 if solved else 0.0
    cost = workflow_latency / budget_total
    utility = quality_reward - λ * cost
    advantage = (utility - mean) / std
    loss = -log_prob * advantage + task_loss + vae_loss * 0.001

Executor training (PPO + Lagrangian):
    quality_reward = 1.0 if solved else 0.0  (episode-level)
    cost = step_latency / budget_total       (step-level contribution)
    reward = quality_reward - λ * cost
    PPO clipped surrogate with K epochs
    Dual update: λ = max(0, λ + lr_λ * (mean_cost - 1))
        Only active after warmup_epochs.
"""

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loguru import logger

from MAR.InfraMind.inframind_router import InfraMindRouter


class InfraMindTrainer:
    """Two-level trainer for InfraMindRouter with PPO-Lagrangian."""

    def __init__(
        self,
        router: InfraMindRouter,
        lr_planner: float = 3e-4,
        lr_executor: float = 3e-4,
        executor_entropy_coef: float = 0.10,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        # PPO hyperparameters
        ppo_epochs: int = 3,
        ppo_clip: float = 0.2,
        # Lagrangian hyperparameters
        lambda_init: float = 0.0,
        lr_lambda: float = 0.001,
        lambda_max: float = 1.0,
        # λ warmup: hold λ=0 for first N epochs so policy learns quality first
        warmup_epochs: int = 3,
    ) -> None:
        self.router = router
        self.executor_entropy_coef = executor_entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # PPO
        self.ppo_epochs = ppo_epochs
        self.ppo_clip = ppo_clip

        # Shared Lagrange multiplier for budget constraint
        self.lambda_lag = lambda_init
        self.lr_lambda = lr_lambda
        self.lambda_max = lambda_max

        # λ warmup: policy learns quality maximization before cost pressure
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0

        # Planner optimizer (MAS modules)
        self.planner_optimizer = torch.optim.Adam(
            router.planner_parameters(), lr=lr_planner,
        )
        # Executor optimizer (MLP backbone + heads)
        self.executor_optimizer = torch.optim.Adam(
            router.executor_parameters(), lr=lr_executor,
        )

        # LR schedulers (attached via attach_lr_schedulers)
        self.planner_scheduler: Optional[ReduceLROnPlateau] = None
        self.executor_scheduler: Optional[ReduceLROnPlateau] = None

    def attach_lr_schedulers(self, patience: int = 2, factor: float = 0.5) -> None:
        """Attach ReduceLROnPlateau schedulers to both optimizers."""
        self.planner_scheduler = ReduceLROnPlateau(
            self.planner_optimizer, mode="max", patience=patience, factor=factor,
        )
        self.executor_scheduler = ReduceLROnPlateau(
            self.executor_optimizer, mode="max", patience=patience, factor=factor,
        )
        logger.info("LR schedulers attached (patience={}, factor={})", patience, factor)

    def step_schedulers(self, val_solve_rate: float) -> None:
        """Step LR schedulers with validation solve rate."""
        if self.planner_scheduler is not None:
            self.planner_scheduler.step(val_solve_rate)
        if self.executor_scheduler is not None:
            self.executor_scheduler.step(val_solve_rate)

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for λ warmup tracking."""
        self._current_epoch = epoch

    @property
    def warmup_active(self) -> bool:
        """True if still in warmup phase (λ held at 0)."""
        return self._current_epoch < self.warmup_epochs

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train_batch(
        self,
        planner_transitions: Iterable[dict],
        executor_transitions: Iterable[dict],
        task_label: int = 0,
    ) -> Dict[str, float]:
        planner_batch: List[dict] = list(planner_transitions)
        executor_batch: List[dict] = list(executor_transitions)
        metrics: Dict[str, float] = {}

        if planner_batch:
            metrics.update(self._train_planner(planner_batch, task_label))
            # Compute cost_ratio for logging
            lats = [float(item.get("workflow_latency_seconds", 0.0)) for item in planner_batch]
            buds = [float(item.get("budget_total", 60.0)) for item in planner_batch]
            ratios = [l / max(b, 1.0) for l, b in zip(lats, buds)]
            metrics["cost_ratio"] = sum(ratios) / len(ratios)
            metrics["constraint_gap"] = metrics["cost_ratio"] - 1.0
        if executor_batch:
            metrics.update(self._train_executor(executor_batch))
        metrics["lambda"] = self.lambda_lag
        return metrics

    # ------------------------------------------------------------------
    # Planner training (REINFORCE + shared Lagrangian)
    # ------------------------------------------------------------------

    def _train_planner(self, batch: List[dict], task_label: int = 0) -> Dict[str, float]:
        """
        REINFORCE with shared Lagrange multiplier:
            quality = 1.0 if solved else 0.0
            cost = workflow_latency / budget_total
            utility = quality - λ * cost
            advantage = (utility - mean) / std
            loss = -log_prob * advantage + task_loss + vae_loss * 0.001
        """
        device = self.router.device

        # Accumulate per-item data across the batch
        log_probs = []
        vae_losses = []
        task_losses = []
        utilities = []

        tasks_y = torch.tensor([task_label], device=device)

        for item in batch:
            # Re-compute forward pass with CURRENT weights (fresh graph)
            fresh = self.router.evaluate_plan(
                query_embedding=item["query_embedding"],
                chosen_topology_idx=item["chosen_topology_idx"],
                chosen_role_indices=item["chosen_role_indices"],
                agent_count=item["agent_count"],
            )
            log_prob = fresh["planner_log_probs"]   # (1, 1) — fresh graph
            vae_loss = fresh["planner_vae_loss"]     # scalar — fresh graph
            task_probs = fresh["task_probs"]          # (1, N_tasks) — fresh graph

            is_solved = float(item.get("is_solved", 0))
            latency = float(item["workflow_latency_seconds"])
            budget = float(item["budget_total"])

            # Lagrangian utility: quality - λ * cost
            quality_reward = 1.0 if is_solved > 0.5 else 0.0
            cost = latency / max(budget, 1e-3)
            effective_lambda = 0.0 if self.warmup_active else self.lambda_lag
            utility = quality_reward - effective_lambda * cost

            utilities.append(utility)
            log_probs.append(log_prob.squeeze())

            # Task classification loss
            task_loss = F.cross_entropy(task_probs, tasks_y)
            task_losses.append(task_loss)

            # VAE regularization
            vae_losses.append(vae_loss)

        # Normalized REINFORCE: advantage = (utility - mean) / std
        baseline = sum(utilities) / max(len(utilities), 1)
        advantages = [u - baseline for u in utilities]
        if len(advantages) > 1:
            adv_std = (sum(a ** 2 for a in advantages) / len(advantages)) ** 0.5
            if adv_std > 1e-8:
                advantages = [a / adv_std for a in advantages]
        answer_losses = [-lp * adv for lp, adv in zip(log_probs, advantages)]
        answer_loss_mean = torch.stack(answer_losses).mean()
        task_loss_mean = torch.stack(task_losses).mean()
        vae_loss_mean = torch.stack(vae_losses).mean()

        loss = answer_loss_mean + task_loss_mean + vae_loss_mean * 0.001

        self.planner_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.router.planner_parameters(), self.max_grad_norm,
        )
        self.planner_optimizer.step()

        avg_utility = sum(utilities) / max(len(utilities), 1)
        return {
            "planner_loss": float(loss.detach().cpu().item()),
            "planner_answer_loss": float(answer_loss_mean.detach().cpu().item()),
            "planner_task_loss": float(task_loss_mean.detach().cpu().item()),
            "planner_vae_loss": float(vae_loss_mean.detach().cpu().item()),
            "planner_avg_utility": avg_utility,
        }

    # ------------------------------------------------------------------
    # Executor training (PPO-Lagrangian)
    # ------------------------------------------------------------------

    def _train_executor(self, batch: List[dict]) -> Dict[str, float]:
        """
        PPO-Lagrangian executor training:
            quality_reward = is_solved (1.0 or 0.0)
            cost = step_latency / budget_total
            reward = quality - λ * cost
            PPO clipped surrogate over K epochs
            Dual update: λ = max(0, λ + lr_λ * (mean_episode_cost - 1))
        """
        device = self.router.device

        states = torch.stack([item["state"]["state"] for item in batch]).to(device)
        joint_idx = torch.tensor(
            [int(item["action"]["joint_index"].item()) for item in batch], device=device,
        )

        # Reconstruct old log-probs from stored rollout probabilities
        old_joint_probs = torch.stack(
            [item["action"]["joint_probs"].squeeze(0) for item in batch]
        ).to(device)
        old_log_probs = torch.log(
            old_joint_probs.gather(1, joint_idx.unsqueeze(-1)).squeeze(-1) + 1e-8
        ).detach()

        # Separate quality and cost signals
        is_solved = torch.tensor(
            [float(item["quality"]) for item in batch], device=device,
        )
        step_latencies = torch.tensor(
            [float(item.get("latency_seconds", 0.0)) for item in batch], device=device,
        )
        budget_total = torch.tensor(
            [float(item.get("budget_total", 60.0)) for item in batch], device=device,
        )

        # Quality reward: binary episode outcome
        quality_reward = is_solved

        # Cost: simple ratio, no urgency multiplier
        cost = step_latencies / budget_total.clamp(min=1.0)

        # Lagrangian reward: quality - λ * cost (λ=0 during warmup)
        effective_lambda = 0.0 if self.warmup_active else self.lambda_lag
        rewards = (quality_reward - effective_lambda * cost).detach()

        # PPO: K epochs on the same batch
        for ppo_epoch in range(self.ppo_epochs):
            action_out = self.router.get_executor_action(states, deterministic=False)
            joint_probs = action_out["joint_probs"]
            log_joint = torch.log(joint_probs + 1e-8)
            new_log_probs = log_joint.gather(1, joint_idx.unsqueeze(-1)).squeeze(-1)

            # Importance ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Advantage: reward - V(s), normalized
            advantage = rewards - action_out["value"].detach()
            if advantage.numel() > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Clipped surrogate objective
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(action_out["value"], rewards)
            entropy = -(joint_probs * log_joint).sum(dim=-1).mean()

            loss = actor_loss + self.value_coef * value_loss - self.executor_entropy_coef * entropy

            self.executor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.router.executor_parameters(), self.max_grad_norm,
            )
            self.executor_optimizer.step()

        # Dual update: adjust λ based on episode-level constraint violation
        # Skipped during warmup — λ stays at 0 so policy learns quality first
        episode_costs = torch.tensor(
            [float(item.get("workflow_latency_seconds", 0.0)) / max(float(item.get("budget_total", 60.0)), 1.0)
             for item in batch], device=device,
        )
        mean_episode_cost = episode_costs.mean().item()
        if not self.warmup_active:
            self.lambda_lag = min(
                self.lambda_max,
                max(0.0, self.lambda_lag + self.lr_lambda * (mean_episode_cost - 1.0)),
            )

        return {
            "executor_loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "executor_entropy": float(entropy.detach().cpu().item()),
            "lambda": self.lambda_lag,
            "mean_cost": mean_episode_cost,
        }
