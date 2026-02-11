#!/usr/bin/env python3
"""Test planner with actual MATH dataset queries to see if embeddings cause collapse."""

import sys
import torch
sys.path.insert(0, '/home/ah872032.ucf/system-aware-mas')

from MAR.InfraMind.inframind_router import SystemAwareRouter
from pathlib import Path

# Load router with checkpoint
checkpoint_path = Path("/blue/qi855292.ucf/ah872032.ucf/checkpoints/inframind/inframind_math.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print("Loading router...")
router = SystemAwareRouter(
    device=torch.device('cpu'),
    role_domain='Math',
)
router.load_state_dict(checkpoint["router_state_dict"])
router.eval()

# Test with actual queries from MATH
test_queries = [
    "In a particular list of three-digit perfect squares, the first perfect square can be turned into each of the others by rearranging its digits. What is the largest number of distinct perfect squares that could be in the list?",
    "Suppose $a$ and $b$ are positive integers, neither of which is a multiple of 3. Find the least possible remainder when $a^2 + b^2$ is divided by 3.",
    "What is $\\frac{-5}{9}\\cdot \\frac{8}{17}$?",
    "Find $\\sin \\frac{4 \\pi}{3}.$",
    "Expand the product ${(2p^2 - 1)(3p^2 + 4)}$.",
]

budgets = [10.0, 30.0, 50.0, 100.0, 200.0]

print("\n=== Testing Planner with Real MATH Queries ===\n")

topologies = ["Chain", "Debate", "Review", "Parallel", "SingleAgent", "Hierarchical"]
topology_counts = {b: {topo: 0 for topo in topologies} for b in budgets}

for budget in budgets:
    print(f"\nBudget = {budget}s:")
    for i, query in enumerate(test_queries):
        plan = router.plan_graph(query, budget_total=budget, deterministic=True, dataset_name="math")
        selected_topology = plan["topology_name"]
        topology_counts[budget][selected_topology] += 1
        print(f"  Query {i+1}: {selected_topology:15s} | role_set={plan['role_set_name']}")

print("\n\n=== Summary: Topology Distribution by Budget ===")
for budget in budgets:
    counts = topology_counts[budget]
    print(f"\nBudget = {budget:5.0f}s:")
    for topo, count in counts.items():
        if count > 0:
            pct = (count / len(test_queries)) * 100
            print(f"  {topo:20s}: {count}/{len(test_queries)} ({pct:5.1f}%)")

# Check if query embeddings are too similar
print("\n\n=== Checking Query Embedding Diversity ===")
embeddings = []
for query in test_queries:
    emb = router.encode_query(query, dataset_name="math")
    embeddings.append(emb)

embeddings_tensor = torch.stack(embeddings)
print(f"Embedding shape: {embeddings_tensor.shape}")
print(f"Mean: {embeddings_tensor.mean().item():.6f}")
print(f"Std:  {embeddings_tensor.std().item():.6f}")

# Compute pairwise cosine similarity
from torch.nn.functional import cosine_similarity
print("\nPairwise cosine similarities:")
for i in range(len(test_queries)):
    for j in range(i+1, len(test_queries)):
        sim = cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
        print(f"  Query {i+1} vs Query {j+1}: {sim:.4f}")
