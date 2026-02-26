"""REINFORCE edge optimizer for GPTSwarm.

Optimizes edge logits using REINFORCE with a moving average baseline.
Only the edge_logits parameter is updated — node operations and models are fixed.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional

import torch

from loguru import logger

from MAR.GPTSwarm.swarm_graph import SwarmGraph


class SwarmTrainer:
    """REINFORCE trainer for GPTSwarm edge probabilities.

    Args:
        graph: The SwarmGraph whose edge_logits will be optimized.
        lr: Learning rate for edge logit updates.
        baseline_decay: Exponential moving average decay for the baseline.
    """

    def __init__(
        self,
        graph: SwarmGraph,
        lr: float = 0.1,
        baseline_decay: float = 0.9,
    ):
        self.graph = graph
        self.baseline_decay = baseline_decay
        self.baseline: float = 0.0
        self.iteration: int = 0
        self.best_val_accuracy: float = 0.0

        # Only optimize edge_logits
        self.optimizer = torch.optim.Adam([graph.edge_logits], lr=lr)

    def train_step(
        self,
        utilities: List[float],
        edge_log_probs_list: List[torch.Tensor],
    ) -> float:
        """Perform one REINFORCE update on edge logits.

        Args:
            utilities: List of scalar utilities (0 or 1) for each batch item.
            edge_log_probs_list: List of summed log_probs tensors from
                graph.forward() for each batch item.

        Returns:
            Mean loss value.
        """
        if not utilities or not edge_log_probs_list:
            return 0.0

        batch_size = len(utilities)

        # Update baseline with EMA
        mean_utility = sum(utilities) / batch_size
        if self.iteration == 0:
            self.baseline = mean_utility
        else:
            self.baseline = (
                self.baseline_decay * self.baseline
                + (1 - self.baseline_decay) * mean_utility
            )

        # REINFORCE loss: -log_probs * (utility - baseline)
        loss = torch.tensor(0.0)
        for utility, log_probs in zip(utilities, edge_log_probs_list):
            if log_probs is not None:
                advantage = utility - self.baseline
                # Ensure log_probs is scalar (sum if needed)
                lp = log_probs.sum() if log_probs.dim() > 0 else log_probs
                loss = loss + (-lp * advantage)
        loss = loss / batch_size

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iteration += 1
        return loss.item()

    def save_checkpoint(self, path: str) -> None:
        """Save trainer state to disk (atomic write)."""
        state = {
            "edge_logits": self.graph.edge_logits.data.clone(),
            "baseline": self.baseline,
            "optimizer_state": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "best_val_accuracy": self.best_val_accuracy,
            "domain": self.graph.domain,
            "node_configs": [
                {"node_id": nc.node_id, "model_name": nc.model_name, "operation": nc.operation}
                for nc in self.graph.node_configs
            ],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        logger.info("[GPTSwarm] Checkpoint saved: {} (iter={})", path, self.iteration)

    def load_checkpoint(self, path: str) -> None:
        """Load trainer state from disk."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.graph.edge_logits.data.copy_(state["edge_logits"])
        self.baseline = state["baseline"]
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.iteration = state["iteration"]
        self.best_val_accuracy = state.get("best_val_accuracy", 0.0)
        logger.info(
            "[GPTSwarm] Checkpoint loaded: {} (iter={}, best_val={:.1%})",
            path, self.iteration, self.best_val_accuracy,
        )
