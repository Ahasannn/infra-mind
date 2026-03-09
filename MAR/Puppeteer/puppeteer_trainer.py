"""REINFORCE trainer for Puppeteer policy.

Optimizes the MLP policy using REINFORCE with a moving average baseline.
Mirrors the SwarmTrainer interface for consistency.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from loguru import logger

from MAR.Puppeteer.puppeteer_policy import PuppeteerPolicy


class PuppeteerTrainer:
    """REINFORCE trainer for Puppeteer sequential agent selection.

    Args:
        policy: The PuppeteerPolicy whose parameters will be optimized.
        lr: Learning rate for policy parameters.
        baseline_decay: Exponential moving average decay for the baseline.
        cost_penalty: Weight for step-count cost penalty in reward.
    """

    def __init__(
        self,
        policy: PuppeteerPolicy,
        lr: float = 1e-3,
        baseline_decay: float = 0.9,
        cost_penalty: float = 0.1,
    ):
        self.policy = policy
        self.baseline_decay = baseline_decay
        self.cost_penalty = cost_penalty
        self.baseline: float = 0.0
        self.iteration: int = 0
        self.best_val_accuracy: float = 0.0

        # Optimize all policy network parameters
        self.optimizer = torch.optim.Adam(policy.policy_net.parameters(), lr=lr)

    def train_step(
        self,
        utilities: List[float],
        log_probs_list: List[List[torch.Tensor]],
        step_counts: List[int],
    ) -> float:
        """Perform one REINFORCE update on policy parameters.

        Args:
            utilities: List of scalar utilities (0 or 1) for each batch item.
            log_probs_list: List of per-item lists of step log_probs tensors.
            step_counts: List of number of steps taken per item.

        Returns:
            Mean loss value.
        """
        if not utilities or not log_probs_list:
            return 0.0

        batch_size = len(utilities)

        # Compute rewards with cost penalty
        rewards = []
        for utility, num_steps in zip(utilities, step_counts):
            cost = (num_steps / self.policy.max_steps) * self.cost_penalty
            rewards.append(utility - cost)

        # Update baseline with EMA
        mean_reward = sum(rewards) / batch_size
        if self.iteration == 0:
            self.baseline = mean_reward
        else:
            self.baseline = (
                self.baseline_decay * self.baseline
                + (1 - self.baseline_decay) * mean_reward
            )

        # REINFORCE loss: -sum(log_probs) * (reward - baseline)
        loss_terms = []
        for reward, step_log_probs in zip(rewards, log_probs_list):
            if step_log_probs:
                advantage = reward - self.baseline
                total_log_prob = torch.stack(step_log_probs).sum()
                loss_terms.append(-total_log_prob * advantage)

        if not loss_terms:
            self.iteration += 1
            return 0.0

        loss = torch.stack(loss_terms).sum() / batch_size

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iteration += 1
        return loss.item()

    def save_checkpoint(self, path: str) -> None:
        """Save trainer state to disk (atomic write)."""
        state = {
            "policy_state_dict": self.policy.policy_net.state_dict(),
            "baseline": self.baseline,
            "optimizer_state": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "best_val_accuracy": self.best_val_accuracy,
            "cost_penalty": self.cost_penalty,
            "domain": self.policy.domain,
            "num_roles": self.policy.num_roles,
            "max_steps": self.policy.max_steps,
            "model_names": self.policy.model_names,
            "agent_roles": self.policy.agent_roles,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        logger.info("[Puppeteer] Checkpoint saved: {} (iter={})", path, self.iteration)

    def load_checkpoint(self, path: str) -> None:
        """Load trainer state from disk."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.policy.policy_net.load_state_dict(state["policy_state_dict"])
        self.baseline = state["baseline"]
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.iteration = state["iteration"]
        self.best_val_accuracy = state.get("best_val_accuracy", 0.0)
        self.cost_penalty = state.get("cost_penalty", self.cost_penalty)
        logger.info(
            "[Puppeteer] Checkpoint loaded: {} (iter={}, best_val={:.1%})",
            path, self.iteration, self.best_val_accuracy,
        )
