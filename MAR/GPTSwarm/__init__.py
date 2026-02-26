"""GPTSwarm baseline — learned inter-agent topology via REINFORCE edge optimization.

Reference: Zhuge et al., "GPTSwarm: Language Agents as Optimizable Graphs", ICML 2024.
"""

from MAR.GPTSwarm.swarm_graph import SwarmGraph, SwarmResult, NodeConfig
from MAR.GPTSwarm.swarm_trainer import SwarmTrainer
from MAR.GPTSwarm.swarm_runner import SwarmRunner
from MAR.GPTSwarm.node_prompts import get_node_prompt, get_aggregation_prompt

__all__ = [
    "SwarmGraph",
    "SwarmResult",
    "NodeConfig",
    "SwarmTrainer",
    "SwarmRunner",
    "get_node_prompt",
    "get_aggregation_prompt",
]
