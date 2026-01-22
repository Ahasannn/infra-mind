import argparse
import csv
import json
import os
import sys
from typing import Iterable, List, Tuple

import torch
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Datasets.mbpp_dataset import MbppDataset


def _select_device(device: str) -> str:
    device = (device or "auto").strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _iter_rows(splits: Iterable[str], field: str) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    for split in splits:
        ds = MbppDataset(split)
        df = ds.df
        if "task_id" not in df.columns:
            raise RuntimeError(f"MBPP split={split} missing task_id column")
        if field not in df.columns:
            raise RuntimeError(f"MBPP split={split} missing {field} column")
        for task_id, text in zip(df["task_id"].tolist(), df[field].tolist()):
            rows.append((int(task_id), str(text)))
    rows.sort(key=lambda item: item[0])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute MBPP query embeddings into a CSV file.")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=("train", "val", "test", "prompt", "all"),
        help="Which MBPP split to embed. Use 'all' to include all splits.",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="task",
        choices=("task", "text"),
        help="Which MBPP column to embed. Training typically uses 'task'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("Datasets", "embeddings", "mbpp_query_embeddings.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for embedding model: auto/cpu/cuda/cuda:0/cuda:1.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit (0 means no limit).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")
    args = parser.parse_args()

    splits = ("train", "val", "test", "prompt") if args.split == "all" else (args.split,)
    rows = _iter_rows(splits, args.field)
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]

    output_path = args.output
    if os.path.exists(output_path) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {output_path} (use --overwrite)")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    device = _select_device(args.device)
    model = SentenceTransformer(args.model, device=device)

    fieldnames = ("query_id", "embedding", "dataset_name")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        batch_size = max(1, int(args.batch_size))
        dataset_name = "mbpp"
        total = len(rows)
        for start in range(0, total, batch_size):
            batch = rows[start : start + batch_size]
            batch_ids = [item[0] for item in batch]
            batch_texts = [item[1] for item in batch]
            embeddings = model.encode(
                batch_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            for query_id, emb in zip(batch_ids, embeddings):
                writer.writerow(
                    {
                        "query_id": query_id,
                        "embedding": json.dumps(emb.tolist()),
                        "dataset_name": dataset_name,
                    }
                )
            print(f"wrote {min(start + batch_size, total)}/{total}", flush=True)

    print(f"done: {output_path}")


if __name__ == "__main__":
    main()
