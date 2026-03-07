"""Inference wrapper for Puppeteer test scripts.

Loads a trained checkpoint and runs deterministic inference.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from loguru import logger

from MAR.Puppeteer.puppeteer_policy import (
    PuppeteerPolicy,
    PuppeteerResult,
    DEFAULT_MODEL_NAMES,
    DEFAULT_AGENT_ROLES,
)


class PuppeteerRunner:
    """Puppeteer inference runner.

    Args:
        model_names: List of model names for the agent pool.
        checkpoint_path: Path to trained checkpoint (policy weights).
        domain: Task domain (mbpp, humaneval, gsm_hard, math, mmlu_pro).
        max_steps: Maximum reasoning chain length.
        max_tokens: Max output tokens per LLM call.
        temperature: Sampling temperature for LLM calls.
        request_timeout: Per-LLM-call timeout in seconds.
        encoder: Pre-initialized SentenceEncoder (shared with caller).
    """

    def __init__(
        self,
        model_names: List[str],
        checkpoint_path: Optional[str] = None,
        domain: str = "mbpp",
        max_steps: int = 5,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        request_timeout: float = 600.0,
        encoder=None,
    ):
        self.model_names = model_names
        num_models = len(model_names)

        self.policy = PuppeteerPolicy(
            num_models=num_models,
            model_names=model_names,
            domain=domain,
            max_steps=max_steps,
            max_tokens=max_tokens,
            temperature=temperature,
            request_timeout=request_timeout,
        )

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.policy.policy_net.load_state_dict(state["policy_state_dict"])
            logger.info(
                "[Puppeteer] Loaded checkpoint: {} (iter={})",
                checkpoint_path, state.get("iteration", "?"),
            )

        # Sentence encoder for query embeddings
        if encoder is not None:
            self.encoder = encoder
        else:
            from MAR.LLM.llm_embedding import SentenceEncoder
            self.encoder = SentenceEncoder()

    def run_single(self, query: str, item_id: str = "") -> PuppeteerResult:
        """Run deterministic inference for a single query.

        Args:
            query: The task/question text.
            item_id: Identifier for telemetry.

        Returns:
            PuppeteerResult with final response and chain details.
        """
        with torch.no_grad():
            query_emb = self.encoder(query).squeeze(0)
            result, _ = self.policy.execute_chain(
                query=query,
                item_id=item_id,
                query_embedding=query_emb,
                deterministic=True,
            )
        return result
