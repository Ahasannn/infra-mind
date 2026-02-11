#!/usr/bin/env python3
"""Check InfraMind router weights and training args."""

import torch
import pprint
from pathlib import Path

checkpoint_path = Path("/blue/qi855292.ucf/ah872032.ucf/checkpoints/inframind/inframind_math.pt")

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print("\n=== Training Args ===")
if "args" in checkpoint:
    args = checkpoint["args"]
    if hasattr(args, "__dict__"):
        args_dict = vars(args)
    else:
        args_dict = args if isinstance(args, dict) else {}
    # Print only relevant args
    relevant_keys = [
        "epochs", "batch_size", "lr", "train_limit", "test_limit",
        "arrival_rate", "budget_total", "checkpoint", "save_checkpoint",
        "dry_run", "inference_only"
    ]
    for key in relevant_keys:
        if key in args_dict:
            print(f"  {key:20s}: {args_dict[key]}")
    # Also print all keys if dict
    print(f"\n  All args keys: {list(args_dict.keys())[:20]}")

print(f"\n=== Epoch: {checkpoint['epoch']} ===")
print(f"Run ID: {checkpoint.get('run_id', 'N/A')}")

# Check router state dict
if "router_state_dict" in checkpoint:
    state_dict = checkpoint["router_state_dict"]

    print("\n=== Router Model Head Weights ===")
    # Check the output heads - these should show if model is biased
    heads = ["model_head.weight", "model_head.bias",
             "strategy_head.weight", "strategy_head.bias",
             "topology_head.weight", "topology_head.bias",
             "role_head.weight", "role_head.bias"]

    for name in heads:
        if name in state_dict:
            param = state_dict[name]
            print(f"\n{name}:")
            print(f"  Shape: {param.shape}")
            print(f"  Mean:  {param.mean().item():+.6f}")
            print(f"  Std:   {param.std().item():.6f}")
            print(f"  Min:   {param.min().item():+.6f}")
            print(f"  Max:   {param.max().item():+.6f}")

            # For biases, show the actual values (small tensors)
            if "bias" in name:
                print(f"  Values: {param.tolist()}")
