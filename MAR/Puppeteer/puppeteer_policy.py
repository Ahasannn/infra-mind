"""Puppeteer MLP policy network with REINFORCE.

Central orchestrator that selects which agent ROLE to activate next
in a sequential reasoning chain. Following the original paper's design,
the policy selects roles only — model assignment is handled separately
(one random model per request, used for all steps in that chain).

State representation re-encodes (query + accumulated chain context) at
each step via sentence-transformer, giving the policy meaningful signal
about previous reasoning outputs.

Architecture:
    Input: [context_emb(384), step_count(1), cost_so_far(1)] = 386 dims
    MLP: 386 -> 512 -> 128 -> 32 -> N_roles (softmax)
    Training: REINFORCE with EMA baseline
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from loguru import logger

from MAR.LLM.gpt_chat import ALLChat
from MAR.Puppeteer.agent_prompts import get_agent_prompt


# Agent roles the policy can select (decoupled from models)
DEFAULT_AGENT_ROLES = ["reasoner", "critic", "specialist", "reflector", "terminator"]

# Default model pool (matches vLLM pool)
DEFAULT_MODEL_NAMES = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

_MAX_CONTEXT_CHARS = 4000


@dataclass
class PuppeteerResult:
    """Result of a single Puppeteer execution chain."""
    query: str = ""
    item_id: str = ""
    final_response: str = ""
    step_responses: Dict[int, str] = field(default_factory=dict)
    step_models: Dict[int, str] = field(default_factory=dict)
    step_roles: Dict[int, str] = field(default_factory=dict)
    step_latencies: Dict[int, float] = field(default_factory=dict)
    step_errors: Dict[int, str] = field(default_factory=dict)
    assigned_model: str = ""
    total_latency: float = 0.0
    num_steps: int = 0

    @property
    def num_steps_succeeded(self) -> int:
        return len(self.step_responses)

    @property
    def num_steps_failed(self) -> int:
        return len(self.step_errors)


class PuppeteerPolicy(nn.Module):
    """MLP policy that selects the next agent ROLE at each step.

    At each step, the policy observes:
      - context_embedding (384-dim): sentence-transformer encoding of
        (query + accumulated chain context), re-computed each step
      - step_count (1-dim, normalized by max_steps)
      - cost_so_far (1-dim, normalized cumulative latency)

    And outputs a categorical distribution over N roles.

    Model assignment is decoupled: one random model is assigned per
    request and used for all steps, following the original paper's
    design of using a single backbone model for all agent roles.

    Args:
        num_roles: Number of agent roles.
        agent_roles: Role labels.
        model_names: List of model names for LLM calls.
        domain: Task domain (mbpp, humaneval, gsm_hard, math, mmlu_pro).
        max_steps: Maximum reasoning chain length.
        embed_dim: Dimension of context embedding.
        max_tokens: Max output tokens per LLM call.
        temperature: Sampling temperature for LLM calls.
        request_timeout: Per-LLM-call timeout in seconds.
    """

    def __init__(
        self,
        num_roles: int = 5,
        agent_roles: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        domain: str = "mbpp",
        max_steps: int = 5,
        embed_dim: int = 384,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        request_timeout: float = 600.0,
    ):
        super().__init__()
        self.num_roles = num_roles
        self.agent_roles = agent_roles or list(DEFAULT_AGENT_ROLES[:num_roles])
        self.model_names = model_names or list(DEFAULT_MODEL_NAMES)
        self.domain = domain
        self.max_steps = max_steps
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout

        input_dim = embed_dim + 2  # context_emb + step_count + cost_so_far

        # MLP policy: 386 -> 512 -> 128 -> 32 -> N_roles
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_roles),
        )

        # Pre-create ALLChat clients (thread-safe)
        self._clients: Dict[str, ALLChat] = {
            name: ALLChat(name) for name in set(self.model_names)
        }

    def forward(
        self,
        context_embedding: torch.Tensor,
        step_count: float,
        cost_so_far: float,
    ) -> Tuple[torch.Tensor, torch.distributions.Categorical]:
        """Compute action distribution for next role selection.

        Args:
            context_embedding: (embed_dim,) tensor — encoding of
                query + accumulated chain context.
            step_count: Current step normalized by max_steps.
            cost_so_far: Normalized cumulative latency.

        Returns:
            logits: Raw policy logits (num_roles,).
            dist: Categorical distribution over roles.
        """
        state = torch.cat([
            context_embedding,
            torch.tensor([step_count, cost_so_far], device=context_embedding.device),
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
        encoder,
        deterministic: bool = False,
        assigned_model: Optional[str] = None,
    ) -> Tuple[PuppeteerResult, List[torch.Tensor]]:
        """Execute a full reasoning chain with sequential role selection.

        At each step:
          1. Re-encode (query + chain context) for dynamic state.
          2. Policy selects next role from state.
          3. Assigned model executes with role-specific prompt.
          4. Terminates at max_steps or when "terminator" role is selected.

        Args:
            query: The task/question text.
            item_id: Identifier for telemetry.
            query_embedding: Pre-computed query embedding (used for step 0).
            encoder: SentenceEncoder for re-encoding context at each step.
            deterministic: If True, use argmax instead of sampling.
            assigned_model: Model to use for all steps. If None, randomly
                selected from model_names.

        Returns:
            result: PuppeteerResult with chain details.
            log_probs: List of log probabilities for each step's action.
        """
        # Assign one random model for this entire chain
        if assigned_model is None:
            assigned_model = random.choice(self.model_names)

        result = PuppeteerResult(
            query=query, item_id=item_id, assigned_model=assigned_model,
        )
        log_probs_list: List[torch.Tensor] = []
        total_start = time.perf_counter()

        chain_context: List[str] = []  # accumulated reasoning outputs
        cumulative_latency = 0.0

        for step in range(self.max_steps):
            step_norm = step / max(self.max_steps - 1, 1)
            cost_norm = min(cumulative_latency / 300.0, 1.0)  # normalize by 5 min

            # Dynamic state: re-encode query + accumulated context
            if step == 0:
                context_emb = query_embedding
            else:
                context_text = query + "\n\n" + "\n\n".join(
                    f"--- Step {i + 1} ---\n{resp[:_MAX_CONTEXT_CHARS]}"
                    for i, resp in enumerate(chain_context)
                )
                with torch.no_grad():
                    context_emb = encoder(context_text).squeeze(0)

            _, dist = self.forward(context_emb, step_norm, cost_norm)

            if deterministic:
                action = dist.probs.argmax()
                log_prob = dist.log_prob(action)
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action)

            log_probs_list.append(log_prob)

            role_idx = action.item()
            agent_role = self.agent_roles[role_idx]

            result.step_models[step] = assigned_model
            result.step_roles[step] = agent_role

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
                response = self._call_model(assigned_model, messages)
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
                    step, assigned_model, agent_role, item_id, exc,
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
