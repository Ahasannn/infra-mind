"""Puppeteer MLP policy network with REINFORCE.

Central orchestrator that selects which agent (model) to activate next
in a sequential reasoning chain. State is encoded via sentence-transformer
embeddings of the query + accumulated reasoning context.

Architecture:
    Input: [query_emb(384), step_count(1), cost_so_far(1)] = 386 dims
    MLP: 386 -> 512 -> 128 -> 32 -> N_models (softmax)
    Training: REINFORCE with EMA baseline
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from loguru import logger

from MAR.LLM.gpt_chat import ALLChat
from MAR.Puppeteer.agent_prompts import get_agent_prompt


# Default model pool (matches vLLM pool)
DEFAULT_MODEL_NAMES = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

# Agent roles assigned to each model position
DEFAULT_AGENT_ROLES = ["reasoner", "critic", "specialist", "reflector", "terminator"]

_MAX_CONTEXT_CHARS = 4000


@dataclass
class PuppeteerResult:
    """Result of a single Puppeteer execution chain."""
    query: str = ""
    item_id: str = ""
    final_response: str = ""
    step_responses: Dict[int, str] = field(default_factory=dict)
    step_models: Dict[int, str] = field(default_factory=dict)
    step_latencies: Dict[int, float] = field(default_factory=dict)
    step_errors: Dict[int, str] = field(default_factory=dict)
    total_latency: float = 0.0
    num_steps: int = 0

    @property
    def num_steps_succeeded(self) -> int:
        return len(self.step_responses)

    @property
    def num_steps_failed(self) -> int:
        return len(self.step_errors)


class PuppeteerPolicy(nn.Module):
    """MLP policy that selects the next agent at each step.

    At each step, the policy observes:
      - query_embedding (384-dim from sentence-transformer)
      - step_count (1-dim, normalized by max_steps)
      - cost_so_far (1-dim, normalized cumulative latency)

    And outputs a categorical distribution over N models.

    Args:
        num_models: Number of agent models in the pool.
        model_names: List of model names for LLM calls.
        agent_roles: Role labels for each model position.
        domain: Task domain (mbpp, humaneval, gsm_hard, math, mmlu_pro).
        max_steps: Maximum reasoning chain length.
        embed_dim: Dimension of query embedding.
        max_tokens: Max output tokens per LLM call.
        temperature: Sampling temperature for LLM calls.
        request_timeout: Per-LLM-call timeout in seconds.
    """

    def __init__(
        self,
        num_models: int = 5,
        model_names: Optional[List[str]] = None,
        agent_roles: Optional[List[str]] = None,
        domain: str = "mbpp",
        max_steps: int = 5,
        embed_dim: int = 384,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        request_timeout: float = 600.0,
    ):
        super().__init__()
        self.num_models = num_models
        self.model_names = model_names or list(DEFAULT_MODEL_NAMES[:num_models])
        self.agent_roles = agent_roles or list(DEFAULT_AGENT_ROLES[:num_models])
        self.domain = domain
        self.max_steps = max_steps
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout

        input_dim = embed_dim + 2  # query_emb + step_count + cost_so_far

        # MLP policy: 386 -> 512 -> 128 -> 32 -> N_models
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_models),
        )

        # Pre-create ALLChat clients (thread-safe)
        self._clients: Dict[str, ALLChat] = {
            name: ALLChat(name) for name in set(self.model_names)
        }

    def forward(
        self,
        query_embedding: torch.Tensor,
        step_count: float,
        cost_so_far: float,
    ) -> Tuple[torch.Tensor, torch.distributions.Categorical]:
        """Compute action distribution for next agent selection.

        Args:
            query_embedding: (embed_dim,) tensor from sentence-transformer.
            step_count: Current step normalized by max_steps.
            cost_so_far: Normalized cumulative latency.

        Returns:
            logits: Raw policy logits (num_models,).
            dist: Categorical distribution over models.
        """
        state = torch.cat([
            query_embedding,
            torch.tensor([step_count, cost_so_far], device=query_embedding.device),
        ])
        logits = self.policy_net(state)
        dist = torch.distributions.Categorical(logits=logits)
        return logits, dist

    def _call_model(self, model_name: str, messages: List[Dict]) -> str:
        """Call a single model's LLM. Thread-safe."""
        client = self._clients[model_name]
        return client.gen(
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            request_timeout=self.request_timeout,
        )

    def execute_chain(
        self,
        query: str,
        item_id: str,
        query_embedding: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[PuppeteerResult, List[torch.Tensor]]:
        """Execute a full reasoning chain with sequential agent selection.

        At each step:
          1. Policy selects next model from state.
          2. Selected model executes with query + previous context.
          3. State updates with new context.
          4. Terminates at max_steps or when "terminator" role is selected.

        Args:
            query: The task/question text.
            item_id: Identifier for telemetry.
            query_embedding: Pre-computed query embedding.
            deterministic: If True, use argmax instead of sampling.

        Returns:
            result: PuppeteerResult with chain details.
            log_probs: List of log probabilities for each step's action.
        """
        result = PuppeteerResult(query=query, item_id=item_id)
        log_probs_list: List[torch.Tensor] = []
        total_start = time.perf_counter()

        chain_context: List[str] = []  # accumulated reasoning outputs
        cumulative_latency = 0.0

        for step in range(self.max_steps):
            step_norm = step / max(self.max_steps - 1, 1)
            cost_norm = min(cumulative_latency / 300.0, 1.0)  # normalize by 5 min

            _, dist = self.forward(query_embedding, step_norm, cost_norm)

            if deterministic:
                action = dist.probs.argmax()
                log_prob = dist.log_prob(action)
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action)

            log_probs_list.append(log_prob)

            model_idx = action.item()
            model_name = self.model_names[model_idx]
            agent_role = self.agent_roles[model_idx]

            result.step_models[step] = model_name

            # Build context from previous steps
            context = None
            if chain_context:
                truncated = []
                for i, text in enumerate(chain_context):
                    t = text[:_MAX_CONTEXT_CHARS]
                    if len(text) > _MAX_CONTEXT_CHARS:
                        t += "\n... [truncated]"
                    truncated.append(f"--- Step {i + 1} Output ---\n{t}")
                context = "\n\n".join(truncated)

            messages = get_agent_prompt(
                domain=self.domain,
                role=agent_role,
                query=query,
                previous_context=context,
                step=step,
                max_steps=self.max_steps,
            )

            step_start = time.perf_counter()
            try:
                response = self._call_model(model_name, messages)
                chain_context.append(response)
                result.step_responses[step] = response
                step_lat = time.perf_counter() - step_start
                result.step_latencies[step] = step_lat
                cumulative_latency += step_lat
            except Exception as exc:
                result.step_errors[step] = str(exc)
                result.step_latencies[step] = time.perf_counter() - step_start
                logger.warning(
                    "[Puppeteer] Step {} ({}/{}) failed for item {}: {}",
                    step, model_name, agent_role, item_id, exc,
                )

            result.num_steps = step + 1

            # Terminate if terminator role is selected
            if agent_role == "terminator":
                break

        # Final response is the last successful step's output
        if result.step_responses:
            last_step = max(result.step_responses.keys())
            result.final_response = result.step_responses[last_step]
        else:
            result.final_response = ""

        result.total_latency = time.perf_counter() - total_start
        return result, log_probs_list
