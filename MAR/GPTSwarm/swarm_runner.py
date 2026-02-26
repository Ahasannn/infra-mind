"""Inference wrapper for GPTSwarm test scripts.

Loads a trained checkpoint and runs deterministic inference.
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from MAR.GPTSwarm.swarm_graph import SwarmGraph, SwarmResult, NodeConfig, DEFAULT_NODE_CONFIGS

import torch


def default_node_configs(model_names: Optional[List[str]] = None) -> List[NodeConfig]:
    """Create the default 5-node config matching the vLLM pool.

    If model_names is provided, maps them positionally to the default
    operation types. Otherwise returns DEFAULT_NODE_CONFIGS.
    """
    if model_names is None:
        return list(DEFAULT_NODE_CONFIGS)

    # Default operations per position
    default_ops = ["cot", "direct", "cot", "direct", "direct"]
    configs = []
    for i, name in enumerate(model_names):
        op = default_ops[i] if i < len(default_ops) else "direct"
        configs.append(NodeConfig(node_id=i, model_name=name, operation=op))
    return configs


class SwarmRunner:
    """GPTSwarm inference runner.

    Args:
        model_names: List of model names for nodes.
        checkpoint_path: Path to trained checkpoint (edge logits).
        domain: Task domain (mbpp, humaneval, gsm8k, math, mmlu).
        aggregator_model: Model name for the aggregator.
        max_tokens: Max output tokens per node call.
        temperature: Sampling temperature for nodes.
        aggregator_temperature: Temperature for the aggregator.
        request_timeout: Per-LLM-call timeout in seconds.
    """

    def __init__(
        self,
        model_names: List[str],
        checkpoint_path: Optional[str] = None,
        domain: str = "mbpp",
        aggregator_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        aggregator_temperature: float = 0.3,
        request_timeout: float = 600.0,
    ):
        node_configs = default_node_configs(model_names)

        self.graph = SwarmGraph(
            node_configs=node_configs,
            aggregator_model=aggregator_model,
            domain=domain,
            max_tokens=max_tokens,
            temperature=temperature,
            aggregator_temperature=aggregator_temperature,
            request_timeout=request_timeout,
        )

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.graph.edge_logits.data.copy_(state["edge_logits"])
            logger.info(
                "[GPTSwarm] Loaded checkpoint: {} (iter={})",
                checkpoint_path, state.get("iteration", "?"),
            )

        # Log edge probabilities
        probs = self.graph.edge_probabilities()
        active = (probs > 0.5).sum().item()
        total = self.graph.edge_mask.sum().item()
        logger.info(
            "[GPTSwarm] Active edges (>0.5): {}/{} possible",
            int(active), int(total),
        )

    def run_single(self, query: str, item_id: str = "") -> SwarmResult:
        """Run deterministic inference for a single query.

        Args:
            query: The task/question text.
            item_id: Identifier for telemetry.

        Returns:
            SwarmResult with final response and node details.
        """
        with torch.no_grad():
            result, _ = self.graph.forward(
                query=query,
                item_id=item_id,
                deterministic=True,
            )
        return result
