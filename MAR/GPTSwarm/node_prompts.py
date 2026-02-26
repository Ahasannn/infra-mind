"""Domain-specific prompts for GPTSwarm node operations.

Three operation types:
- "direct": Answer immediately (IO node).
- "cot": Chain-of-thought reasoning before answering.
- "reflect": Review predecessor outputs and improve/correct.

Aggregation prompts reuse the same domain-aware format constraints.
"""

from typing import Dict, List, Optional

_MAX_CONTEXT_CHARS = 4000


def _format_context(predecessor_outputs: List[str]) -> str:
    """Format predecessor node outputs for reflection nodes."""
    parts = []
    for i, text in enumerate(predecessor_outputs, 1):
        truncated = text[:_MAX_CONTEXT_CHARS]
        if len(text) > _MAX_CONTEXT_CHARS:
            truncated += "\n... [truncated]"
        parts.append(f"--- Node {i} Output ---\n{truncated}")
    return "\n\n".join(parts)


# ── Code domain (MBPP, HumanEval) ────────────────────────────────────────────

def _code_direct(query: str, context: Optional[str]) -> List[Dict[str, str]]:
    msgs = [
        {
            "role": "system",
            "content": (
                "You are an expert programmer. Solve the given task directly.\n\n"
                "IMPORTANT: Your response MUST contain exactly one ```python``` code block "
                "with the solution. Do NOT include test cases or assertions."
            ),
        },
    ]
    if context:
        msgs.append({
            "role": "user",
            "content": f"## Task\n{query}\n\n## Previous Attempts\n{context}\n\nProvide your solution in a ```python``` code block.",
        })
    else:
        msgs.append({
            "role": "user",
            "content": f"## Task\n{query}\n\nProvide your solution in a ```python``` code block.",
        })
    return msgs


def _code_cot(query: str, context: Optional[str]) -> List[Dict[str, str]]:
    msgs = [
        {
            "role": "system",
            "content": (
                "You are an expert programmer. Think step-by-step about the problem, "
                "consider edge cases, then provide a solution.\n\n"
                "IMPORTANT: Your response MUST contain exactly one ```python``` code block "
                "with the final solution. Do NOT include test cases or assertions."
            ),
        },
    ]
    if context:
        msgs.append({
            "role": "user",
            "content": (
                f"## Task\n{query}\n\n## Previous Attempts\n{context}\n\n"
                "Think step-by-step, then provide your solution in a ```python``` code block."
            ),
        })
    else:
        msgs.append({
            "role": "user",
            "content": (
                f"## Task\n{query}\n\n"
                "Think step-by-step, then provide your solution in a ```python``` code block."
            ),
        })
    return msgs


def _code_reflect(query: str, context: Optional[str]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert code reviewer. You will be given a programming task "
                "and outputs from other agents. Review their solutions for correctness, "
                "edge cases, and efficiency. Provide an improved solution.\n\n"
                "IMPORTANT: Your response MUST contain exactly one ```python``` code block "
                "with the improved solution. Do NOT include test cases or assertions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Task\n{query}\n\n## Agent Outputs to Review\n{context or 'No outputs available.'}\n\n"
                "Review the above solutions, identify any issues, and provide the best "
                "solution in a ```python``` code block."
            ),
        },
    ]


# ── Math domain (GSM8K, MATH) ────────────────────────────────────────────────

def _math_direct(query: str, context: Optional[str]) -> List[Dict[str, str]]:
    msgs = [
        {
            "role": "system",
            "content": (
                "You are an expert mathematician. Solve the given problem.\n\n"
                "IMPORTANT: End your response with exactly: The answer is [answer]"
            ),
        },
    ]
    if context:
        msgs.append({
            "role": "user",
            "content": f"## Problem\n{query}\n\n## Previous Attempts\n{context}\n\nEnd with: The answer is [answer]",
        })
    else:
        msgs.append({
            "role": "user",
            "content": f"## Problem\n{query}\n\nEnd with: The answer is [answer]",
        })
    return msgs


def _math_cot(query: str, context: Optional[str]) -> List[Dict[str, str]]:
    msgs = [
        {
            "role": "system",
            "content": (
                "You are an expert mathematician. Think through the problem step-by-step, "
                "showing all your work.\n\n"
                "IMPORTANT: End your response with exactly: The answer is [answer]"
            ),
        },
    ]
    if context:
        msgs.append({
            "role": "user",
            "content": (
                f"## Problem\n{query}\n\n## Previous Attempts\n{context}\n\n"
                "Think step-by-step, then end with: The answer is [answer]"
            ),
        })
    else:
        msgs.append({
            "role": "user",
            "content": (
                f"## Problem\n{query}\n\n"
                "Think step-by-step, then end with: The answer is [answer]"
            ),
        })
    return msgs


def _math_reflect(query: str, context: Optional[str]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert math reviewer. You will be given a math problem "
                "and solutions from other agents. Check their reasoning for errors "
                "and provide the correct answer.\n\n"
                "IMPORTANT: End your response with exactly: The answer is [answer]"
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Problem\n{query}\n\n## Agent Solutions to Review\n{context or 'No solutions available.'}\n\n"
                "Check the reasoning, correct any errors, and end with: The answer is [answer]"
            ),
        },
    ]


# ── MMLU domain ───────────────────────────────────────────────────────────────

def _mmlu_direct(query: str, context: Optional[str]) -> List[Dict[str, str]]:
    msgs = [
        {
            "role": "system",
            "content": (
                "You are an expert at answering multiple-choice questions. "
                "Answer the question.\n\n"
                "IMPORTANT: Your final answer MUST be exactly one letter: A, B, C, or D."
            ),
        },
    ]
    if context:
        msgs.append({
            "role": "user",
            "content": f"## Question\n{query}\n\n## Previous Answers\n{context}\n\nAnswer with a single letter: A, B, C, or D.",
        })
    else:
        msgs.append({
            "role": "user",
            "content": f"## Question\n{query}\n\nAnswer with a single letter: A, B, C, or D.",
        })
    return msgs


def _mmlu_cot(query: str, context: Optional[str]) -> List[Dict[str, str]]:
    msgs = [
        {
            "role": "system",
            "content": (
                "You are an expert at answering multiple-choice questions. "
                "Think through each option carefully before answering.\n\n"
                "IMPORTANT: Your final answer MUST be exactly one letter: A, B, C, or D."
            ),
        },
    ]
    if context:
        msgs.append({
            "role": "user",
            "content": (
                f"## Question\n{query}\n\n## Previous Answers\n{context}\n\n"
                "Think step-by-step, then answer with a single letter: A, B, C, or D."
            ),
        })
    else:
        msgs.append({
            "role": "user",
            "content": (
                f"## Question\n{query}\n\n"
                "Think step-by-step, then answer with a single letter: A, B, C, or D."
            ),
        })
    return msgs


def _mmlu_reflect(query: str, context: Optional[str]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert reviewer for multiple-choice questions. "
                "Review other agents' answers and reasoning, then determine "
                "the correct answer.\n\n"
                "IMPORTANT: Your final answer MUST be exactly one letter: A, B, C, or D."
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Question\n{query}\n\n## Agent Answers to Review\n{context or 'No answers available.'}\n\n"
                "Review the reasoning, then answer with a single letter: A, B, C, or D."
            ),
        },
    ]


# ── Aggregation prompts (same format as MoA) ─────────────────────────────────

def _code_aggregate(query: str, node_outputs: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert code aggregator. You will be given a programming task "
                "and multiple candidate solutions from different agents. Your job is to "
                "select or synthesize the best solution.\n\n"
                "IMPORTANT: Your response MUST contain exactly one ```python``` code block "
                "with the final solution. Do NOT include any test cases or assertions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Task\n{query}\n\n"
                f"## Agent Solutions\n{_format_context(node_outputs)}\n\n"
                "Analyze the solutions, identify the best approach, and output "
                "the final solution inside a single ```python``` code block."
            ),
        },
    ]


def _math_aggregate(query: str, node_outputs: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert math aggregator. You will be given a math problem "
                "and multiple candidate answers from different agents. Check the reasoning "
                "of each response and determine the correct answer.\n\n"
                "IMPORTANT: End your response with exactly: The answer is [answer]"
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Problem\n{query}\n\n"
                f"## Agent Answers\n{_format_context(node_outputs)}\n\n"
                "Analyze each agent's reasoning, identify errors if any, and provide "
                "the correct final answer. End with: The answer is [answer]"
            ),
        },
    ]


def _mmlu_aggregate(query: str, node_outputs: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert aggregator for multiple-choice questions. You will be "
                "given a question with options and multiple candidate answers from different "
                "agents. Determine the correct answer by majority vote or reasoning.\n\n"
                "IMPORTANT: Your final answer MUST be exactly one letter: A, B, C, or D."
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Question\n{query}\n\n"
                f"## Agent Answers\n{_format_context(node_outputs)}\n\n"
                "Based on the agents above, output the correct answer as a single "
                "letter: A, B, C, or D."
            ),
        },
    ]


# ── Public API ────────────────────────────────────────────────────────────────

# Operation → domain → prompt builder
_OPERATION_MAP = {
    "direct": {"code": _code_direct, "math": _math_direct, "mmlu": _mmlu_direct},
    "cot": {"code": _code_cot, "math": _math_cot, "mmlu": _mmlu_cot},
    "reflect": {"code": _code_reflect, "math": _math_reflect, "mmlu": _mmlu_reflect},
}

# Domain aliases
_DOMAIN_ALIASES = {
    "mbpp": "code",
    "humaneval": "code",
    "gsm8k": "math",
    "math": "math",
    "mmlu": "mmlu",
}

_AGGREGATE_MAP = {
    "code": _code_aggregate,
    "math": _math_aggregate,
    "mmlu": _mmlu_aggregate,
}


def get_node_prompt(
    domain: str,
    operation: str,
    query: str,
    predecessor_outputs: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Build messages for a swarm node.

    Args:
        domain: One of mbpp, humaneval, gsm8k, math, mmlu.
        operation: One of direct, cot, reflect.
        query: The original task/question text.
        predecessor_outputs: Outputs from connected upstream nodes (for reflect).

    Returns:
        OpenAI-style message list.
    """
    canon = _DOMAIN_ALIASES.get(domain, domain)
    ops = _OPERATION_MAP.get(operation)
    if ops is None:
        raise ValueError(f"Unknown operation '{operation}'. Expected: {list(_OPERATION_MAP)}")
    fn = ops.get(canon)
    if fn is None:
        raise ValueError(f"Unknown domain '{domain}'. Expected: {list(_DOMAIN_ALIASES)}")

    context = None
    if predecessor_outputs:
        context = _format_context(predecessor_outputs)
    return fn(query, context)


def get_aggregation_prompt(
    domain: str,
    query: str,
    node_outputs: List[str],
) -> List[Dict[str, str]]:
    """Build aggregation messages for the final decision node.

    Args:
        domain: One of mbpp, humaneval, gsm8k, math, mmlu.
        query: The original task/question text.
        node_outputs: List of node output strings.

    Returns:
        OpenAI-style message list for the aggregator call.
    """
    canon = _DOMAIN_ALIASES.get(domain, domain)
    fn = _AGGREGATE_MAP.get(canon)
    if fn is None:
        raise ValueError(f"Unknown domain '{domain}'. Expected: {list(_DOMAIN_ALIASES)}")
    return fn(query, node_outputs)
