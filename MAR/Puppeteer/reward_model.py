"""DeepSeek-32B reward model for Puppeteer training.

Faithful to the original Puppeteer paper which uses a large LLM as a
reward model to score intermediate reasoning quality.  We use
DeepSeek-R1-Distill-Qwen-32B (already in the vLLM pool) instead of the
paper's 70B model.

Only used during training — inference (testing) relies on deterministic
argmax policy without reward model.

Note: DeepSeek-R1 is a reasoning model that generates chain-of-thought
before the final answer.  We parse the score from the LAST number in
the response (after the reasoning), and use max_tokens=512 to ensure
the full reasoning + score fits.
"""

from __future__ import annotations

import re
from typing import Optional

from loguru import logger

from MAR.LLM.gpt_chat import ALLChat


# Domain-specific evaluation prompts
# The "Output ONLY the integer score on the LAST line" instruction helps
# DeepSeek place the score at the end after its chain-of-thought.
_REWARD_PROMPTS = {
    "code": (
        "You are an expert code evaluator.  Given a programming task and a "
        "candidate solution, rate the solution quality on a scale of 0 to 10.\n\n"
        "Criteria:\n"
        "- Correctness: Does the code solve the task?\n"
        "- Completeness: Are edge cases handled?\n"
        "- Clarity: Is the code readable and well-structured?\n\n"
        "You may think step-by-step, but you MUST end your response with "
        "exactly one line containing only the score as: Score: N"
    ),
    "math": (
        "You are an expert math evaluator.  Given a math problem and a "
        "candidate solution, rate the solution quality on a scale of 0 to 10.\n\n"
        "Criteria:\n"
        "- Correctness: Is the final answer correct?\n"
        "- Reasoning: Are the steps logically sound?\n"
        "- Completeness: Is the solution complete?\n\n"
        "You may think step-by-step, but you MUST end your response with "
        "exactly one line containing only the score as: Score: N"
    ),
    "mmlu": (
        "You are an expert evaluator for multiple-choice questions.  Given a "
        "question and a candidate answer, rate the answer quality on a scale "
        "of 0 to 10.\n\n"
        "Criteria:\n"
        "- Correctness: Is the chosen option correct?\n"
        "- Reasoning: Is the justification sound?\n"
        "- Confidence: Does the answer show clear understanding?\n\n"
        "You may think step-by-step, but you MUST end your response with "
        "exactly one line containing only the score as: Score: N"
    ),
}

# Map dataset names to domain keys
_DOMAIN_MAP = {
    "mbpp": "code",
    "humaneval": "code",
    "gsm8k": "math",
    "gsm_hard": "math",
    "math": "math",
    "mmlu": "mmlu",
    "mmlu_pro": "mmlu",
}

# Regex to find "Score: N" pattern (preferred, at end of response)
_SCORE_TAG_RE = re.compile(r"[Ss]core\s*:\s*(\d{1,2})")
# Fallback: find the last standalone integer in the response
_LAST_INT_RE = re.compile(r"\b(\d{1,2})\b")

# Default reward model
DEFAULT_REWARD_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"


class PuppeteerRewardModel:
    """LLM-based reward model for scoring reasoning chain outputs.

    Uses DeepSeek-32B to evaluate the quality of each chain's final
    response on a continuous 0-1 scale.  This provides richer gradient
    signal than binary utility for REINFORCE training.

    Args:
        model_name: vLLM model name for the reward model.
        domain: Dataset domain (mbpp, humaneval, gsm_hard, math, mmlu_pro).
        max_tokens: Max tokens for the reward model response.
        temperature: Sampling temperature (low for deterministic scoring).
        request_timeout: Timeout for reward model calls.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_REWARD_MODEL,
        domain: str = "math",
        max_tokens: int = 512,
        temperature: float = 0.1,
        request_timeout: float = 300.0,
    ):
        self.model_name = model_name
        self.domain = domain
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout

        canon = _DOMAIN_MAP.get(domain, domain)
        self.system_prompt = _REWARD_PROMPTS.get(canon)
        if self.system_prompt is None:
            raise ValueError(
                f"Unknown domain '{domain}' for reward model. "
                f"Expected one of: {list(_DOMAIN_MAP)}"
            )

        self._client = ALLChat(model_name)
        logger.info(
            "[RewardModel] Initialized with model={}, domain={}",
            model_name, domain,
        )

    def score(self, query: str, response: str) -> float:
        """Score a chain output on a 0-1 scale.

        Args:
            query: The original task/question.
            response: The chain's final response text.

        Returns:
            Float score in [0, 1].  Returns 0.0 on failure.
        """
        if not response or not response.strip():
            return 0.0

        # Truncate very long responses to avoid wasting reward model tokens
        truncated_response = response[:4000]

        user_msg = (
            f"## Task\n{query}\n\n"
            f"## Candidate Solution\n{truncated_response}\n\n"
            f"Rate this solution from 0 to 10.  End with: Score: N"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

        try:
            raw = self._client.gen(
                messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                request_timeout=self.request_timeout,
            )
            return self._parse_score(raw)
        except Exception as exc:
            logger.warning(
                "[RewardModel] Scoring failed: {}. Returning 0.0", exc
            )
            return 0.0

    @staticmethod
    def _parse_score(raw: str) -> float:
        """Extract score from model response and normalize to [0, 1].

        Strategy:
        1. Look for explicit "Score: N" pattern (most reliable).
        2. Fallback: take the LAST integer (0-10) in the response,
           since DeepSeek-R1 puts reasoning first and answer last.
        """
        text = raw.strip()

        # Strategy 1: Look for "Score: N" pattern
        matches = list(_SCORE_TAG_RE.finditer(text))
        if matches:
            val = int(matches[-1].group(1))  # take the last match
            return min(max(val / 10.0, 0.0), 1.0)

        # Strategy 2: Take the last standalone integer (0-10)
        all_ints = list(_LAST_INT_RE.finditer(text))
        if all_ints:
            # Filter to 0-10 range, take the last one
            for m in reversed(all_ints):
                val = int(m.group(1))
                if 0 <= val <= 10:
                    return val / 10.0

        logger.warning(
            "[RewardModel] Could not parse score from: '{}'. Returning 0.0",
            text[-200:],
        )
        return 0.0
