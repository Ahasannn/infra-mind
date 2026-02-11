"""
InfraMind Trainer — two-level training for hierarchical CMDP router.

Planner training (MAS-style REINFORCE):
    utility = is_solved - (latency / budget) * latency_rate
    loss = -log_prob * utility + task_loss + vae_loss * 0.001

Executor training (CMDP Actor-Critic):
    reward = quality - λ · (step_latency / budget_remaining)
    loss = actor_loss + value_coef * value_loss - entropy_coef * entropy
    λ updated via signed Lagrangian dual ascent.
"""

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

from MAR.InfraMind.inframind_router import InfraMindRouter


class InfraMindTrainer:
    """Two-level trainer for InfraMindRouter."""

    def __init__(
        self,
        router: InfraMindRouter,
        lr_planner: float = 3e-4,
        lr_executor: float = 3e-4,
        latency_rate: float = 1.0,
        executor_entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        lambda_lr: float = 1e-3,
        mini_batch_size: int = 32,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.router = router
        self.latency_rate = latency_rate
        self.executor_entropy_coef = executor_entropy_coef
        self.value_coef = value_coef
        self.lambda_lr = lambda_lr
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm

        # Planner optimizer (MAS modules + BCFM)
        self.planner_optimizer = torch.optim.Adam(
            router.planner_parameters(), lr=lr_planner,
        )
        # Executor optimizer (MLP backbone + heads)
        self.executor_optimizer = torch.optim.Adam(
            router.executor_parameters(), lr=lr_executor,
        )

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
        if executor_batch:
            metrics.update(self._train_executor(executor_batch))
        return metrics

    # ------------------------------------------------------------------
    # Planner training (MAS-style REINFORCE)
    # ------------------------------------------------------------------

    def _train_planner(self, batch: List[dict], task_label: int = 0) -> Dict[str, float]:
        """
        MAS-style REINFORCE with latency cost:
            utility = is_solved - (latency / budget) * latency_rate
            answer_loss = -log_prob * utility
            loss = answer_loss + task_loss + vae_loss * 0.001
        """
        device = self.router.device

        # Accumulate losses across the batch
        answer_losses = []
        vae_losses = []
        task_losses = []
        utilities = []

        tasks_y = torch.tensor([task_label], device=device)

        for item in batch:
            log_prob = item["planner_log_probs"]          # (1, 1) — differentiable
            vae_loss = item["planner_vae_loss"]            # scalar — differentiable
            task_probs = item["task_probs"]                # (1, N_tasks) — differentiable
            is_solved = float(item.get("is_solved", 0))
            latency = float(item["workflow_latency_seconds"])
            budget = float(item["budget_total"])

            # Utility: is_solved ∈ {0,1} minus normalized latency cost
            cost = (latency / max(budget, 1e-3)) * self.latency_rate
            utility = is_solved - cost
            utilities.append(utility)

            # REINFORCE loss
            answer_loss = -log_prob.squeeze() * utility
            answer_losses.append(answer_loss)

            # Task classification loss
            task_loss = F.cross_entropy(task_probs, tasks_y)
            task_losses.append(task_loss)

            # VAE regularization
            vae_losses.append(vae_loss)

        # Average losses
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
    # Executor training (CMDP Actor-Critic)
    # ------------------------------------------------------------------

    def _train_executor(self, batch: List[dict]) -> Dict[str, float]:
        device = self.router.device

        states = torch.stack([item["state"]["state"] for item in batch]).to(device)
        model_idx = torch.tensor(
            [int(item["action"]["model_index"].item()) for item in batch], device=device,
        )
        strategy_idx = torch.tensor(
            [int(item["action"]["strategy_index"].item()) for item in batch], device=device,
        )
        qualities = torch.tensor([item["quality"] for item in batch], device=device)
        latencies = torch.tensor(
            [float(item.get("workflow_latency_seconds", item.get("latency_seconds", 0.0)))
             for item in batch], device=device,
        )
        budgets = torch.tensor([item["budget_remaining"] for item in batch], device=device)
        rewards = self.router.compute_executor_reward(qualities, latencies, budgets)

        # Mini-batch SGD
        n = len(batch)
        bs = min(self.mini_batch_size, n)
        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_updates = 0

        indices = torch.randperm(n, device=device)
        for start in range(0, n, bs):
            mb_idx = indices[start: start + bs]
            mb_states = states[mb_idx]
            mb_model = model_idx[mb_idx]
            mb_strat = strategy_idx[mb_idx]
            mb_rewards = rewards[mb_idx]

            action_out = self.router.get_executor_action(mb_states, deterministic=False)
            log_model = torch.log(action_out["model_probs"] + 1e-8)
            log_strategy = torch.log(action_out["strategy_probs"] + 1e-8)
            chosen_log_prob = (
                log_model.gather(1, mb_model.unsqueeze(-1)).squeeze(-1)
                + log_strategy.gather(1, mb_strat.unsqueeze(-1)).squeeze(-1)
            )
            entropy = (
                -(action_out["model_probs"] * torch.log(action_out["model_probs"] + 1e-8)).sum(dim=1)
                + -(action_out["strategy_probs"] * torch.log(action_out["strategy_probs"] + 1e-8)).sum(dim=1)
            )
            advantage = mb_rewards - action_out["value"].detach()
            actor_loss = -(chosen_log_prob * advantage).mean()
            value_loss = F.mse_loss(action_out["value"], mb_rewards)
            loss = actor_loss + self.value_coef * value_loss - self.executor_entropy_coef * entropy.mean()

            self.executor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.router.executor_parameters(), self.max_grad_norm,
            )
            self.executor_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_updates += 1

        # Signed Lagrange dual update
        with torch.no_grad():
            cost_ratio = (latencies / budgets.clamp_min(1e-3)).mean()
            constraint_gap = cost_ratio - 1.0
            self.router.lagrange_multiplier.add_(self.lambda_lr * constraint_gap)
            self.router.lagrange_multiplier.clamp_(-5.0, 10.0)

        denom = max(num_updates, 1)
        return {
            "executor_loss": total_loss / denom,
            "actor_loss": total_actor_loss / denom,
            "value_loss": total_value_loss / denom,
            "constraint_gap": float(constraint_gap.item()),
            "cost_ratio": float(cost_ratio.item()),
            "lambda": float(F.softplus(self.router.lagrange_multiplier).item()),
            "executor_entropy": float(entropy.mean().detach().cpu().item()),
        }
