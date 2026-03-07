"""Puppeteer baseline — centralized RL orchestrator for sequential agent chaining.

Reference: "Multi-Agent Collaboration via Evolving Orchestration" (NeurIPS 2025, OpenBMB).
Adapted: 70B reward model replaced with sentence-transformer (384-dim),
         GPT-4o agents replaced with 5 local vLLM models.
"""

from MAR.Puppeteer.puppeteer_policy import PuppeteerPolicy, PuppeteerResult
from MAR.Puppeteer.puppeteer_trainer import PuppeteerTrainer
from MAR.Puppeteer.puppeteer_runner import PuppeteerRunner
from MAR.Puppeteer.agent_prompts import get_agent_prompt

__all__ = [
    "PuppeteerPolicy",
    "PuppeteerResult",
    "PuppeteerTrainer",
    "PuppeteerRunner",
    "get_agent_prompt",
]
