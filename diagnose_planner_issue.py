#!/usr/bin/env python3
"""Diagnose why planner always selects Chain topology."""

import torch
import torch.nn.functional as F
from pathlib import Path

checkpoint_path = Path("/blue/qi855292.ucf/ah872032.ucf/checkpoints/inframind/inframind_math.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint["router_state_dict"]

# Get topology biases
topology_bias = state_dict["topology_head.bias"]
topologies = ["Chain", "Debate", "Review", "Parallel", "SingleAgent", "Hierarchical"]

print("=== Planner Topology Selection Analysis ===\n")
print("Topology head biases:")
for i, topo in enumerate(topologies):
    print(f"  [{i}] {topo:20s}: {topology_bias[i].item():+.6f}")

# Softmax with only bias (assuming hidden state is zero on average)
probs_bias_only = F.softmax(topology_bias, dim=0)
print("\nSoftmax probabilities (bias only, zero hidden state):")
for i, topo in enumerate(topologies):
    print(f"  [{i}] {topo:20s}: {probs_bias_only[i].item()*100:.2f}%")

# Check the topology head weights
topology_weight = state_dict["topology_head.weight"]
print(f"\n\nTopology head weight shape: {topology_weight.shape}")
print(f"Weight statistics:")
print(f"  Mean: {topology_weight.mean().item():+.6f}")
print(f"  Std:  {topology_weight.std().item():.6f}")
print(f"  Min:  {topology_weight.min().item():+.6f}")
print(f"  Max:  {topology_weight.max().item():+.6f}")

# Check per-topology weight norms
print("\nPer-topology weight L2 norms:")
for i, topo in enumerate(topologies):
    weight_norm = torch.norm(topology_weight[i]).item()
    print(f"  [{i}] {topo:20s}: {weight_norm:.6f}")

# Simulate with a few different query embeddings to see if topology changes
print("\n\n=== Simulating with different query embeddings ===")

# Load the router to simulate
import sys
sys.path.insert(0, '/home/ah872032.ucf/system-aware-mas')
from MAR.InfraMind.inframind_router import SystemAwareRouter

print("Loading router...")
router = SystemAwareRouter(
    device=torch.device('cpu'),
    role_domain='Math',
)
router.load_state_dict(state_dict)
router.eval()

# Test with different budgets and random queries
budgets = [10.0, 30.0, 50.0, 100.0, 200.0]
print("\nTesting topology selection with different budgets:")

for budget in budgets:
    # Create random query embeddings
    topology_counts = {topo: 0 for topo in topologies}

    for trial in range(50):
        query_emb = torch.randn(router.embedding_dim)
        planner_state = torch.cat([query_emb, torch.tensor([budget])], dim=0).unsqueeze(0)

        hidden = router.planner_backbone(planner_state)
        topo_logits = router.topology_head(hidden)
        topo_probs = F.softmax(topo_logits, dim=-1)

        # Argmax (deterministic)
        topo_idx = topo_probs.argmax(dim=-1).item()
        topology_counts[topologies[topo_idx]] += 1

    print(f"\n  Budget={budget:5.0f}s:")
    for topo, count in topology_counts.items():
        pct = (count / 50) * 100
        print(f"    {topo:20s}: {count:2d}/50 ({pct:5.1f}%)")
