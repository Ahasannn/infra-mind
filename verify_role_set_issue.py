#!/usr/bin/env python3
"""Verify that multi-agent role sets are broken for Math domain."""
import sys
sys.path.insert(0, '/home/ah872032.ucf/system-aware-mas')
from MAR.Utils.utils import get_kwargs

# Simulate what happens with different topologies and N=1 agent
topologies = ["Chain", "Debate", "FullConnected", "Reflection", "CoT", "IO"]
for topo in topologies:
    kwargs = get_kwargs(topo, 1)  # 1 agent
    spatial = kwargs.get("fixed_spatial_masks", [])
    temporal = kwargs.get("fixed_temporal_masks", [])
    num_rounds = kwargs.get("num_rounds", 1)
    print(f"{topo:15s} (N=1): spatial={spatial}, temporal={temporal}, rounds={num_rounds}")

print("\n--- Now with N=3 agents ---")
for topo in topologies:
    kwargs = get_kwargs(topo, 3)
    spatial = kwargs.get("fixed_spatial_masks", [])
    temporal = kwargs.get("fixed_temporal_masks", [])
    num_rounds = kwargs.get("num_rounds", 1)
    print(f"{topo:15s} (N=3): rounds={num_rounds}")
    print(f"  spatial:  {spatial}")
    print(f"  temporal: {temporal}")
