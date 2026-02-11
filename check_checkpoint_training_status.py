#!/usr/bin/env python3
"""Check if InfraMind checkpoint is trained or random."""

import torch
from pathlib import Path

checkpoint_path = Path("/blue/qi855292.ucf/ah872032.ucf/checkpoints/inframind/inframind_math.pt")

if not checkpoint_path.exists():
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    exit(1)

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print("\n=== Checkpoint Keys ===")
for key in checkpoint.keys():
    print(f"  - {key}")

if "metadata" in checkpoint:
    meta = checkpoint["metadata"]
    print("\n=== Metadata ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")

# Check if model weights look random (small variance, near zero mean) or trained (larger variance)
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]

    print("\n=== Model Weight Statistics ===")
    for name, param in state_dict.items():
        if "weight" in name and param.numel() > 0:
            mean = param.mean().item()
            std = param.std().item()
            absmax = param.abs().max().item()
            print(f"  {name:50s} | mean={mean:+.6f} std={std:.6f} max={absmax:.6f}")

# Check if there are any training epochs
if "epoch" in checkpoint:
    print(f"\n✅ Trained for {checkpoint['epoch']} epochs")
else:
    print("\n⚠️  No epoch information - might be untrained")

if "train_episodes" in checkpoint.get("metadata", {}):
    print(f"✅ Trained on {checkpoint['metadata']['train_episodes']} episodes")
else:
    print("⚠️  No training episode info")
