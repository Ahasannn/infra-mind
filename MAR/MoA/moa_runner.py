"""MoA (Mixture of Agents) runner.

Layer 1: All proposer models answer in parallel.
Layer 2: Aggregator synthesizes the best answer from all proposals.

Thread-safe — designed to be called from RequestShooter handlers.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

from loguru import logger

from MAR.LLM.gpt_chat import ALLChat
from MAR.MoA.aggregation_prompts import get_aggregation_prompt


@dataclass
class MoAResult:
    """Result of a single MoA invocation."""

    query: str
    item_id: str
    proposer_responses: Dict[str, str] = field(default_factory=dict)
    proposer_latencies: Dict[str, float] = field(default_factory=dict)
    proposer_errors: Dict[str, str] = field(default_factory=dict)
    aggregated_response: str = ""
    aggregator_latency: float = 0.0
    total_latency: float = 0.0

    @property
    def num_succeeded(self) -> int:
        return len(self.proposer_responses)

    @property
    def num_failed(self) -> int:
        return len(self.proposer_errors)

    @property
    def proposer_latency_max(self) -> float:
        return max(self.proposer_latencies.values()) if self.proposer_latencies else 0.0

    @property
    def proposer_latency_min(self) -> float:
        return min(self.proposer_latencies.values()) if self.proposer_latencies else 0.0


class MoARunner:
    """Mixture of Agents: query all models, then aggregate.

    Args:
        model_names: List of proposer model names (matches vLLM profile).
        aggregator_model: Model name for the aggregator (default: strongest).
        domain: Task domain for aggregation prompt selection.
        max_tokens: Max output tokens for proposers.
        temperature: Proposer temperature.
        aggregator_temperature: Aggregator temperature (lower for more deterministic).
        request_timeout: Per-LLM-call timeout in seconds.
    """

    def __init__(
        self,
        model_names: List[str],
        aggregator_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        domain: str = "mbpp",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        aggregator_temperature: float = 0.3,
        request_timeout: float = 600.0,
    ):
        self.model_names = model_names
        self.aggregator_model = aggregator_model
        self.domain = domain
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.aggregator_temperature = aggregator_temperature
        self.request_timeout = request_timeout

        # Pre-create ALLChat clients (lightweight, thread-safe via httpx connection pool)
        self._clients: Dict[str, ALLChat] = {
            name: ALLChat(name) for name in model_names
        }
        self._aggregator_client = ALLChat(aggregator_model)

    def _call_proposer(self, model_name: str, messages: List[Dict]) -> str:
        """Call a single proposer model. Runs in a thread."""
        client = self._clients[model_name]
        return client.gen(
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            request_timeout=self.request_timeout,
        )

    def run_single(
        self,
        query: str,
        item_id: str = "",
        system_prompt: Optional[str] = None,
        aggregation_prompt_fn: Optional[Callable] = None,
    ) -> MoAResult:
        """Run MoA for a single query.

        Args:
            query: The task/question text.
            item_id: Identifier for telemetry.
            system_prompt: Optional system prompt for proposers.
            aggregation_prompt_fn: Custom aggregation prompt builder.
                Signature: (domain, query, proposals) -> messages.
                Defaults to get_aggregation_prompt.

        Returns:
            MoAResult with all proposer and aggregator data.
        """
        result = MoAResult(query=query, item_id=item_id)
        total_start = time.perf_counter()

        # --- Layer 1: Parallel proposer calls ---
        proposer_messages = []
        if system_prompt:
            proposer_messages.append({"role": "system", "content": system_prompt})
        proposer_messages.append({"role": "user", "content": query})

        futures = {}
        with ThreadPoolExecutor(max_workers=len(self.model_names)) as pool:
            for name in self.model_names:
                fut = pool.submit(self._call_proposer, name, proposer_messages)
                futures[fut] = name

            for fut in as_completed(futures):
                name = futures[fut]
                t0 = time.perf_counter()
                try:
                    resp = fut.result()
                    latency = time.perf_counter() - total_start  # wall time from start
                    result.proposer_responses[name] = resp
                    result.proposer_latencies[name] = latency
                except Exception as exc:
                    latency = time.perf_counter() - total_start
                    result.proposer_errors[name] = str(exc)
                    result.proposer_latencies[name] = latency
                    logger.warning(
                        "[MoA] Proposer {} failed for item {}: {}",
                        name, item_id, exc,
                    )

        if not result.proposer_responses:
            result.aggregated_response = ""
            result.total_latency = time.perf_counter() - total_start
            logger.error("[MoA] All proposers failed for item {}", item_id)
            return result

        # --- Layer 2: Aggregation ---
        proposals = list(result.proposer_responses.values())
        prompt_fn = aggregation_prompt_fn or get_aggregation_prompt
        agg_messages = prompt_fn(self.domain, query, proposals)

        agg_start = time.perf_counter()
        try:
            result.aggregated_response = self._aggregator_client.gen(
                agg_messages,
                max_tokens=self.max_tokens,
                temperature=self.aggregator_temperature,
                request_timeout=self.request_timeout,
            )
        except Exception as exc:
            logger.error("[MoA] Aggregator failed for item {}: {}", item_id, exc)
            result.aggregated_response = ""
        result.aggregator_latency = time.perf_counter() - agg_start

        result.total_latency = time.perf_counter() - total_start
        return result
