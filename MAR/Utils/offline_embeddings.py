import csv
import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch


def load_mbpp_query_embeddings(
    path: Union[str, Path],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[int, torch.Tensor]:
    """
    Load MBPP query embeddings from `Datasets/embeddings/mbpp_query_embeddings.csv`.

    Expected columns:
      - query_id: MBPP task_id (int)
      - embedding: JSON list of floats (384-d for MiniLM-L6-v2)
      - dataset_name: (optional) should be 'mbpp'
    """
    if not path:
        return {}

    path_str = str(path)
    if not os.path.isfile(path_str):
        return {}

    embeddings: Dict[int, torch.Tensor] = {}
    with open(path_str, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = row.get("query_id")
            raw_emb = row.get("embedding")
            if raw_id is None or raw_emb is None:
                continue
            try:
                query_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            try:
                values = json.loads(raw_emb)
            except json.JSONDecodeError:
                continue
            if not isinstance(values, list) or not values:
                continue
            try:
                tensor = torch.tensor(values, device=device, dtype=dtype)
            except (TypeError, ValueError):
                continue
            embeddings[query_id] = tensor
    return embeddings


def load_role_embeddings(
    path: Union[str, Path],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Load role embeddings from `Datasets/embeddings/role_embeddings.csv`.

    Expected columns:
      - role_name: role identifier string
      - embedding: JSON list of floats (384-d for MiniLM-L6-v2)
      - dataset_name: (optional) should be 'roles_code'
    """
    if not path:
        return {}

    path_str = str(path)
    if not os.path.isfile(path_str):
        return {}

    embeddings: Dict[str, torch.Tensor] = {}
    with open(path_str, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row.get("role_name")
            raw_emb = row.get("embedding")
            if raw_name is None or raw_emb is None:
                continue
            role_name = str(raw_name).strip()
            if not role_name:
                continue
            try:
                values = json.loads(raw_emb)
            except json.JSONDecodeError:
                continue
            if not isinstance(values, list) or not values:
                continue
            try:
                tensor = torch.tensor(values, device=device, dtype=dtype)
            except (TypeError, ValueError):
                continue
            embeddings[role_name] = tensor
    return embeddings

