"""Agent-type-specific prompts for Puppeteer sequential chain.

Five agent roles that the policy can select at each step:
- "reasoner": Primary problem solver with detailed reasoning.
- "critic": Reviews and critiques previous reasoning.
- "specialist": Domain expert focused on technical aspects.
- "reflector": Reflects on and synthesizes prior outputs.
- "terminator": Produces final concise answer from accumulated context.

Each role is adapted per domain (code, math, mmlu).
"""

from typing import Dict, List, Optional


# ── Domain aliases ────────────────────────────────────────────────────────────

_DOMAIN_MAP = {
    "mbpp": "code",
    "humaneval": "code",
    "gsm8k": "math",
    "gsm_hard": "math",
    "math": "math",
    "mmlu": "mmlu",
    "mmlu_pro": "mmlu",
}


# ── Code domain (MBPP, HumanEval) ────────────────────────────────────────────

_CODE_PROMPTS = {
    "reasoner": {
        "system": (
            "You are an expert programmer and primary problem solver. "
            "Analyze the task carefully, consider edge cases, and provide a complete solution.\n\n"
            "IMPORTANT: Your response MUST contain exactly one ```python``` code block "
            "with the solution. Do NOT include test cases or assertions."
        ),
        "user_first": "## Task\n{query}\n\nProvide your solution in a ```python``` code block.",
        "user_chain": (
            "## Task\n{query}\n\n## Previous Reasoning\n{context}\n\n"
            "Build on or improve the above reasoning. Provide your solution in a ```python``` code block."
        ),
    },
    "critic": {
        "system": (
            "You are an expert code reviewer. Review previous solutions for bugs, "
            "edge cases, and logical errors. Provide a corrected solution.\n\n"
            "IMPORTANT: Your response MUST contain exactly one ```python``` code block "
            "with the corrected solution. Do NOT include test cases or assertions."
        ),
        "user_first": "## Task\n{query}\n\nProvide your solution in a ```python``` code block.",
        "user_chain": (
            "## Task\n{query}\n\n## Solutions to Review\n{context}\n\n"
            "Identify any bugs or issues and provide a corrected solution in a ```python``` code block."
        ),
    },
    "specialist": {
        "system": (
            "You are a specialist programmer with deep knowledge of algorithms and data structures. "
            "Focus on correctness and efficiency.\n\n"
            "IMPORTANT: Your response MUST contain exactly one ```python``` code block "
            "with the solution. Do NOT include test cases or assertions."
        ),
        "user_first": "## Task\n{query}\n\nProvide your solution in a ```python``` code block.",
        "user_chain": (
            "## Task\n{query}\n\n## Context\n{context}\n\n"
            "As a specialist, provide an optimized solution in a ```python``` code block."
        ),
    },
    "reflector": {
        "system": (
            "You are a reflective programmer. Synthesize insights from previous attempts "
            "and produce an improved solution.\n\n"
            "IMPORTANT: Your response MUST contain exactly one ```python``` code block "
            "with the improved solution. Do NOT include test cases or assertions."
        ),
        "user_first": "## Task\n{query}\n\nProvide your solution in a ```python``` code block.",
        "user_chain": (
            "## Task\n{query}\n\n## Previous Attempts\n{context}\n\n"
            "Reflect on what worked and what didn't. Provide an improved solution in a ```python``` code block."
        ),
    },
    "terminator": {
        "system": (
            "You are a final-answer synthesizer. You have access to all previous reasoning. "
            "Select or synthesize the best solution and present it clearly.\n\n"
            "IMPORTANT: Your response MUST contain exactly one ```python``` code block "
            "with the final solution. Do NOT include test cases or assertions."
        ),
        "user_first": "## Task\n{query}\n\nProvide the final solution in a ```python``` code block.",
        "user_chain": (
            "## Task\n{query}\n\n## All Previous Reasoning\n{context}\n\n"
            "Select or synthesize the best solution. Output the final answer in a ```python``` code block."
        ),
    },
}


# ── Math domain (GSM-Hard, MATH) ─────────────────────────────────────────────

_MATH_PROMPTS = {
    "reasoner": {
        "system": (
            "You are an expert mathematician. Think through the problem step-by-step.\n\n"
            "IMPORTANT: End your response with exactly: The answer is [answer]"
        ),
        "user_first": "## Problem\n{query}\n\nThink step-by-step, then end with: The answer is [answer]",
        "user_chain": (
            "## Problem\n{query}\n\n## Previous Reasoning\n{context}\n\n"
            "Build on the above reasoning. End with: The answer is [answer]"
        ),
    },
    "critic": {
        "system": (
            "You are a math critic. Check previous solutions for calculation errors "
            "and logical mistakes.\n\n"
            "IMPORTANT: End your response with exactly: The answer is [answer]"
        ),
        "user_first": "## Problem\n{query}\n\nEnd with: The answer is [answer]",
        "user_chain": (
            "## Problem\n{query}\n\n## Solutions to Check\n{context}\n\n"
            "Verify the reasoning and correct any errors. End with: The answer is [answer]"
        ),
    },
    "specialist": {
        "system": (
            "You are a math specialist with expertise in advanced techniques. "
            "Apply the most appropriate mathematical method.\n\n"
            "IMPORTANT: End your response with exactly: The answer is [answer]"
        ),
        "user_first": "## Problem\n{query}\n\nEnd with: The answer is [answer]",
        "user_chain": (
            "## Problem\n{query}\n\n## Context\n{context}\n\n"
            "Apply your expertise. End with: The answer is [answer]"
        ),
    },
    "reflector": {
        "system": (
            "You are a reflective mathematician. Synthesize previous attempts "
            "and identify the correct approach.\n\n"
            "IMPORTANT: End your response with exactly: The answer is [answer]"
        ),
        "user_first": "## Problem\n{query}\n\nEnd with: The answer is [answer]",
        "user_chain": (
            "## Problem\n{query}\n\n## Previous Attempts\n{context}\n\n"
            "Reflect and synthesize. End with: The answer is [answer]"
        ),
    },
    "terminator": {
        "system": (
            "You are a final-answer judge. Review all previous reasoning and "
            "determine the correct answer.\n\n"
            "IMPORTANT: End your response with exactly: The answer is [answer]"
        ),
        "user_first": "## Problem\n{query}\n\nEnd with: The answer is [answer]",
        "user_chain": (
            "## Problem\n{query}\n\n## All Previous Reasoning\n{context}\n\n"
            "Determine the correct final answer. End with: The answer is [answer]"
        ),
    },
}


# ── MMLU domain ───────────────────────────────────────────────────────────────

_MMLU_PROMPTS = {
    "reasoner": {
        "system": (
            "You are an expert at answering multiple-choice questions. "
            "Think through each option carefully.\n\n"
            "IMPORTANT: Your final answer MUST be exactly one letter: A, B, C, D, E, F, G, H, I, or J."
        ),
        "user_first": "## Question\n{query}\n\nThink step-by-step, then answer with a single letter.",
        "user_chain": (
            "## Question\n{query}\n\n## Previous Reasoning\n{context}\n\n"
            "Build on the above. Answer with a single letter."
        ),
    },
    "critic": {
        "system": (
            "You are a critical reviewer of multiple-choice answers. "
            "Check previous reasoning for errors.\n\n"
            "IMPORTANT: Your final answer MUST be exactly one letter: A, B, C, D, E, F, G, H, I, or J."
        ),
        "user_first": "## Question\n{query}\n\nAnswer with a single letter.",
        "user_chain": (
            "## Question\n{query}\n\n## Answers to Review\n{context}\n\n"
            "Check the reasoning and provide the correct letter answer."
        ),
    },
    "specialist": {
        "system": (
            "You are a domain specialist. Apply your expertise to this question.\n\n"
            "IMPORTANT: Your final answer MUST be exactly one letter: A, B, C, D, E, F, G, H, I, or J."
        ),
        "user_first": "## Question\n{query}\n\nAnswer with a single letter.",
        "user_chain": (
            "## Question\n{query}\n\n## Context\n{context}\n\n"
            "Apply your expertise. Answer with a single letter."
        ),
    },
    "reflector": {
        "system": (
            "You are a reflective thinker. Synthesize previous answers "
            "and reasoning to find the best answer.\n\n"
            "IMPORTANT: Your final answer MUST be exactly one letter: A, B, C, D, E, F, G, H, I, or J."
        ),
        "user_first": "## Question\n{query}\n\nAnswer with a single letter.",
        "user_chain": (
            "## Question\n{query}\n\n## Previous Answers\n{context}\n\n"
            "Reflect and determine the best answer. Answer with a single letter."
        ),
    },
    "terminator": {
        "system": (
            "You are a final-answer synthesizer. Determine the correct answer "
            "from all accumulated reasoning.\n\n"
            "IMPORTANT: Your final answer MUST be exactly one letter: A, B, C, D, E, F, G, H, I, or J."
        ),
        "user_first": "## Question\n{query}\n\nAnswer with a single letter.",
        "user_chain": (
            "## Question\n{query}\n\n## All Previous Reasoning\n{context}\n\n"
            "Determine the final answer. Answer with a single letter."
        ),
    },
}


# ── Dispatch map ──────────────────────────────────────────────────────────────

_PROMPT_MAP = {
    "code": _CODE_PROMPTS,
    "math": _MATH_PROMPTS,
    "mmlu": _MMLU_PROMPTS,
}


def get_agent_prompt(
    domain: str,
    role: str,
    query: str,
    previous_context: Optional[str] = None,
    step: int = 0,
    max_steps: int = 5,
) -> List[Dict[str, str]]:
    """Build messages for a Puppeteer agent at a given step.

    Args:
        domain: One of mbpp, humaneval, gsm_hard, gsm8k, math, mmlu, mmlu_pro.
        role: One of reasoner, critic, specialist, reflector, terminator.
        query: The original task/question text.
        previous_context: Formatted outputs from previous steps (None for step 0).
        step: Current step index.
        max_steps: Maximum chain length.

    Returns:
        OpenAI-style message list.
    """
    canon = _DOMAIN_MAP.get(domain, domain)
    domain_prompts = _PROMPT_MAP.get(canon)
    if domain_prompts is None:
        raise ValueError(f"Unknown domain '{domain}'. Expected: {list(_DOMAIN_MAP)}")

    role_prompts = domain_prompts.get(role)
    if role_prompts is None:
        raise ValueError(f"Unknown role '{role}'. Expected: {list(domain_prompts)}")

    system_msg = role_prompts["system"]

    if previous_context:
        user_msg = role_prompts["user_chain"].format(query=query, context=previous_context)
    else:
        user_msg = role_prompts["user_first"].format(query=query)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
