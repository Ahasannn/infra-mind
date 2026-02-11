#!/usr/bin/env python3
"""Check the final Lagrange multiplier value."""
import torch
import torch.nn.functional as F

checkpoint = torch.load("/blue/qi855292.ucf/ah872032.ucf/checkpoints/inframind/inframind_math.pt", map_location="cpu")
state = checkpoint["router_state_dict"]
lam_raw = state["lagrange_multiplier"]
lam_softplus = F.softplus(lam_raw)

print(f"Lagrange multiplier (raw param): {lam_raw.item():.6f}")
print(f"Lagrange multiplier (softplus):  {lam_softplus.item():.6f}")
print()

# Simulate rewards for different quality/latency combos
print("=== Reward Simulation: quality - lambda * penalty ===")
print(f"Lambda = {lam_softplus.item():.4f}")
print()

budgets = [10, 20, 50, 100, 200]
latencies = [3, 5, 10, 20, 50, 100]
quality = 1.0  # perfect quality

print(f"{'Budget':>8s} | {'Latency':>8s} | {'Over-budget':>12s} | {'Penalty':>10s} | {'Reward (q=1.0)':>15s} | {'Reward (q=0.0)':>15s}")
print("-" * 85)
for b in budgets:
    for lat in latencies:
        over = max(0, lat - b)
        penalty = over / b if b > 0 else 0
        reward_q1 = 1.0 - lam_softplus.item() * penalty
        reward_q0 = 0.0 - lam_softplus.item() * penalty
        marker = " <-- OVER" if lat > b else ""
        print(f"{b:>8d} | {lat:>8d} | {over:>12d} | {penalty:>10.4f} | {reward_q1:>15.4f} | {reward_q0:>15.4f}{marker}")
    print()
