"""Domain-specific aggregation prompts for MoA.

Each prompt instructs the aggregator to synthesize multiple proposer responses
into a single answer that matches the expected output format for evaluation.
"""

from typing import Dict, List

_MAX_PROPOSAL_CHARS = 4000


def _format_proposals(proposals: List[str]) -> str:
    parts = []
    for i, p in enumerate(proposals, 1):
        text = p[:_MAX_PROPOSAL_CHARS]
        if len(p) > _MAX_PROPOSAL_CHARS:
            text += "\n... [truncated]"
        parts.append(f"--- Response {i} ---\n{text}")
    return "\n\n".join(parts)


def _code_prompt(query: str, proposals: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert code aggregator. You will be given a programming task "
                "and multiple candidate solutions from different models. Your job is to "
                "select or synthesize the best solution.\n\n"
                "IMPORTANT: Your response MUST contain exactly one ```python``` code block "
                "with the final solution. Do NOT include any test cases or assertions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Task\n{query}\n\n"
                f"## Candidate Solutions\n{_format_proposals(proposals)}\n\n"
                "Analyze the candidate solutions, identify the best approach, and output "
                "the final solution inside a single ```python``` code block."
            ),
        },
    ]


def _math_prompt(query: str, proposals: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert math aggregator. You will be given a math problem "
                "and multiple candidate answers from different models. Check the reasoning "
                "of each response and determine the correct answer.\n\n"
                "IMPORTANT: End your response with exactly: The answer is [answer]"
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Problem\n{query}\n\n"
                f"## Candidate Answers\n{_format_proposals(proposals)}\n\n"
                "Analyze each candidate's reasoning, identify errors if any, and provide "
                "the correct final answer. End with: The answer is [answer]"
            ),
        },
    ]


def _mmlu_prompt(query: str, proposals: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert aggregator for multiple-choice questions. You will be "
                "given a question with options and multiple candidate answers from different "
                "models. Determine the correct answer by majority vote or reasoning.\n\n"
                "IMPORTANT: Your final answer MUST be exactly one letter: A, B, C, or D."
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Question\n{query}\n\n"
                f"## Candidate Answers\n{_format_proposals(proposals)}\n\n"
                "Based on the candidates above, output the correct answer as a single "
                "letter: A, B, C, or D."
            ),
        },
    ]


_DOMAIN_MAP = {
    "mbpp": _code_prompt,
    "humaneval": _code_prompt,
    "gsm8k": _math_prompt,
    "math": _math_prompt,
    "mmlu": _mmlu_prompt,
}


def get_aggregation_prompt(
    domain: str, query: str, proposals: List[str]
) -> List[Dict[str, str]]:
    """Build aggregation messages for the given domain.

    Args:
        domain: One of mbpp, humaneval, gsm8k, math, mmlu.
        query: The original task/question text.
        proposals: List of proposer response strings.

    Returns:
        OpenAI-style message list for the aggregator call.
    """
    fn = _DOMAIN_MAP.get(domain)
    if fn is None:
        raise ValueError(f"Unknown domain '{domain}'. Expected one of: {list(_DOMAIN_MAP)}")
    return fn(query, proposals)
