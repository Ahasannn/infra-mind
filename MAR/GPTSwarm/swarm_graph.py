"""GPTSwarm computational graph with learnable edges.

Core of the GPTSwarm baseline (ICML 2024 Oral). Implements:
- Fixed nodes (one per model, each with an operation type).
- Learnable edge logits (upper-triangular DAG, no self-loops).
- Stochastic edge sampling (Bernoulli) for REINFORCE training.
- Deterministic thresholding (>0.5) for test-time inference.
- Sequential node execution in index order (lower → higher).
- Final aggregation by a fixed aggregator model.

Thread-safe — designed to be called from RequestShooter handlers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from loguru import logger

from MAR.LLM.gpt_chat import ALLChat
from MAR.GPTSwarm.node_prompts import get_node_prompt, get_aggregation_prompt


@dataclass
class NodeConfig:
    """Configuration for a single swarm node."""
    node_id: int
    model_name: str
    operation: str  # "direct", "cot", or "reflect"


@dataclass
class SwarmResult:
    """Result of a single SwarmGraph invocation."""
    query: str = ""
    item_id: str = ""
    final_response: str = ""
    node_responses: Dict[int, str] = field(default_factory=dict)
    node_latencies: Dict[int, float] = field(default_factory=dict)
    node_errors: Dict[int, str] = field(default_factory=dict)
    realized_edges: List[Tuple[int, int]] = field(default_factory=list)
    total_latency: float = 0.0
    aggregator_latency: float = 0.0

    @property
    def num_active_edges(self) -> int:
        return len(self.realized_edges)

    @property
    def num_nodes_succeeded(self) -> int:
        return len(self.node_responses)

    @property
    def num_nodes_failed(self) -> int:
        return len(self.node_errors)


# Default 5-node topology (one per model in the vLLM pool)
DEFAULT_NODE_CONFIGS = [
    NodeConfig(0, "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "cot"),
    NodeConfig(1, "mistralai/Mistral-Small-24B-Instruct-2501", "direct"),
    NodeConfig(2, "Qwen/Qwen2.5-Coder-14B-Instruct", "cot"),
    NodeConfig(3, "meta-llama/Llama-3.1-8B-Instruct", "direct"),
    NodeConfig(4, "meta-llama/Llama-3.2-3B-Instruct", "direct"),
]


class SwarmGraph(nn.Module):
    """Computational graph with learnable edge probabilities.

    Nodes execute sequentially in index order (0→N-1). Each node receives
    context from connected predecessor nodes (lower-indexed nodes with
    active edges). Edges are sampled via Bernoulli(sigmoid(logits)) during
    training and thresholded at 0.5 during inference.

    Args:
        node_configs: List of NodeConfig defining nodes.
        aggregator_model: Model name for the final aggregation step.
        domain: Task domain (mbpp, humaneval, gsm8k, math, mmlu).
        max_tokens: Max output tokens per node call.
        temperature: Sampling temperature for node calls.
        aggregator_temperature: Temperature for the aggregator.
        request_timeout: Per-LLM-call timeout in seconds.
    """

    def __init__(
        self,
        node_configs: Optional[List[NodeConfig]] = None,
        aggregator_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        domain: str = "mbpp",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        aggregator_temperature: float = 0.3,
        request_timeout: float = 600.0,
    ):
        super().__init__()
        self.node_configs = node_configs or list(DEFAULT_NODE_CONFIGS)
        self.aggregator_model = aggregator_model
        self.domain = domain
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.aggregator_temperature = aggregator_temperature
        self.request_timeout = request_timeout

        N = len(self.node_configs)

        # Learnable edge logits: N x N, initialized to 0 (sigmoid → 0.5)
        self.edge_logits = nn.Parameter(torch.zeros(N, N))

        # Mask: upper-triangular only (no self-loops, no backward edges)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        self.register_buffer("edge_mask", mask)

        # Pre-create ALLChat clients (thread-safe)
        unique_models = set(nc.model_name for nc in self.node_configs)
        unique_models.add(aggregator_model)
        self._clients: Dict[str, ALLChat] = {
            name: ALLChat(name) for name in unique_models
        }

    @property
    def num_nodes(self) -> int:
        return len(self.node_configs)

    def edge_probabilities(self) -> torch.Tensor:
        """Return edge probability matrix (masked, no grad needed for inspection)."""
        probs = torch.sigmoid(self.edge_logits)
        return probs * self.edge_mask.float()

    def realize_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample binary adjacency matrix via Bernoulli.

        Returns:
            adjacency: Binary N x N tensor (1 = active edge).
            log_probs: Log probabilities of the sampled edges (for REINFORCE).
        """
        probs = torch.sigmoid(self.edge_logits)
        probs = probs * self.edge_mask.float()  # zero out non-upper-triangular

        # Bernoulli sampling
        dist = torch.distributions.Bernoulli(probs=probs)
        adjacency = dist.sample()
        log_probs = dist.log_prob(adjacency)

        # Only count log_probs for valid (upper-triangular) positions
        log_probs = log_probs * self.edge_mask.float()

        return adjacency, log_probs

    def realize_graph_deterministic(self) -> torch.Tensor:
        """Threshold edge probabilities at 0.5 for deterministic inference.

        Returns:
            adjacency: Binary N x N tensor.
        """
        probs = torch.sigmoid(self.edge_logits)
        probs = probs * self.edge_mask.float()
        return (probs >= 0.5).float()

    def _call_node(self, node: NodeConfig, messages: List[Dict]) -> str:
        """Call a single node's LLM. Thread-safe."""
        client = self._clients[node.model_name]
        return client.gen(
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            request_timeout=self.request_timeout,
        )

    def forward(
        self,
        query: str,
        item_id: str = "",
        deterministic: bool = False,
    ) -> Tuple[SwarmResult, Optional[torch.Tensor]]:
        """Execute the swarm graph for a single query.

        Args:
            query: The task/question text.
            item_id: Identifier for telemetry.
            deterministic: If True, use threshold at 0.5 instead of sampling.

        Returns:
            result: SwarmResult with all node outputs and aggregated response.
            edge_log_probs: Sum of log probs for sampled edges (None if deterministic).
        """
        result = SwarmResult(query=query, item_id=item_id)
        total_start = time.perf_counter()

        # Realize edges
        if deterministic:
            adjacency = self.realize_graph_deterministic()
            edge_log_probs_sum = None
        else:
            adjacency, log_probs = self.realize_graph()
            edge_log_probs_sum = log_probs.sum()

        # Record realized edges
        N = self.num_nodes
        for i in range(N):
            for j in range(N):
                if adjacency[i, j] > 0.5:
                    result.realized_edges.append((i, j))

        # Execute nodes in index order (sequential DAG execution)
        node_outputs: Dict[int, str] = {}
        for idx, node in enumerate(self.node_configs):
            # Collect outputs from predecessor nodes with active edges
            predecessors = []
            for pred_idx in range(idx):
                if adjacency[pred_idx, idx] > 0.5 and pred_idx in node_outputs:
                    predecessors.append(node_outputs[pred_idx])

            # Build messages for this node
            messages = get_node_prompt(
                domain=self.domain,
                operation=node.operation,
                query=query,
                predecessor_outputs=predecessors if predecessors else None,
            )

            # Call the node
            node_start = time.perf_counter()
            try:
                response = self._call_node(node, messages)
                node_outputs[idx] = response
                result.node_responses[idx] = response
                result.node_latencies[idx] = time.perf_counter() - node_start
            except Exception as exc:
                result.node_errors[idx] = str(exc)
                result.node_latencies[idx] = time.perf_counter() - node_start
                logger.warning(
                    "[GPTSwarm] Node {} ({}) failed for item {}: {}",
                    idx, node.model_name, item_id, exc,
                )

        # Aggregation step
        if node_outputs:
            outputs_list = [node_outputs[i] for i in sorted(node_outputs.keys())]
            agg_messages = get_aggregation_prompt(self.domain, query, outputs_list)

            agg_start = time.perf_counter()
            try:
                agg_client = self._clients[self.aggregator_model]
                result.final_response = agg_client.gen(
                    agg_messages,
                    max_tokens=self.max_tokens,
                    temperature=self.aggregator_temperature,
                    request_timeout=self.request_timeout,
                )
            except Exception as exc:
                logger.error("[GPTSwarm] Aggregator failed for item {}: {}", item_id, exc)
                result.final_response = ""
            result.aggregator_latency = time.perf_counter() - agg_start
        else:
            logger.error("[GPTSwarm] All nodes failed for item {}", item_id)
            result.final_response = ""

        result.total_latency = time.perf_counter() - total_start
        return result, edge_log_probs_sum
