"""MoA (Mixture of Agents) baseline.

Always queries ALL models, then aggregates — infrastructure-blind brute-force.
Zero training required (inference only).
"""

from MAR.MoA.moa_runner import MoARunner, MoAResult
from MAR.MoA.aggregation_prompts import get_aggregation_prompt

__all__ = ["MoARunner", "MoAResult", "get_aggregation_prompt"]
