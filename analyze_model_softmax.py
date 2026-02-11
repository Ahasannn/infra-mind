#!/usr/bin/env python3
"""Analyze model selection softmax from checkpoint biases."""

import torch
import torch.nn.functional as F

checkpoint_path = "/blue/qi855292.ucf/ah872032.ucf/checkpoints/inframind/inframind_math.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

state_dict = checkpoint["router_state_dict"]
model_bias = state_dict["model_head.bias"]

models = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

print("=== Model Selection Bias Analysis ===\n")
print("Model head biases (before softmax):")
for i, model in enumerate(models):
    print(f"  [{i}] {model:50s}: {model_bias[i].item():+.6f}")

# Simulate softmax with zero hidden state (just bias)
print("\n\nSoftmax probabilities with ZERO hidden state (only bias):")
probs = F.softmax(model_bias, dim=0)
for i, model in enumerate(models):
    print(f"  [{i}] {model:50s}: {probs[i].item()*100:.2f}%")

# Check topology biases
topology_bias = state_dict["topology_head.bias"]
topologies = ["Chain", "Debate", "Review", "Parallel", "SingleAgent", "Hierarchical"]  # Guessing names
print("\n\n=== Topology Selection Bias Analysis ===\n")
print("Topology head biases:")
for i in range(len(topology_bias)):
    topo_name = topologies[i] if i < len(topologies) else f"Topo{i}"
    print(f"  [{i}] {topo_name:20s}: {topology_bias[i].item():+.6f}")

topo_probs = F.softmax(topology_bias, dim=0)
print("\nSoftmax probabilities (only bias):")
for i in range(len(topology_bias)):
    topo_name = topologies[i] if i < len(topologies) else f"Topo{i}"
    print(f"  [{i}] {topo_name:20s}: {topo_probs[i].item()*100:.2f}%")

# Check strategy biases
strategy_bias = state_dict["strategy_head.bias"]
strategies = ["Flash", "Concise", "DeepThink"]
print("\n\n=== Strategy Selection Bias Analysis ===\n")
print("Strategy head biases:")
for i, strat in enumerate(strategies):
    print(f"  [{i}] {strat:20s}: {strategy_bias[i].item():+.6f}")

strat_probs = F.softmax(strategy_bias, dim=0)
print("\nSoftmax probabilities (only bias):")
for i, strat in enumerate(strategies):
    print(f"  [{i}] {strat:20s}: {strat_probs[i].item()*100:.2f}%")
