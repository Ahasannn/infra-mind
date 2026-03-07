"""MMLU-Pro dataset loader.

Loads from HuggingFace: TIGER-Lab/MMLU-Pro
10 answer choices (A-J), 14 categories, test (12032) + validation (70) splits.

Split mapping:
  - 'train': Stratified sample from HF 'test' split (first N per category)
  - 'test':  Stratified sample from HF 'test' split (next N per category, non-overlapping with train)
  - 'val'/'dev': All 70 items from HF 'validation' split
"""

import string
from typing import Union, List, Literal, Any, Dict, Optional

import numpy as np
from abc import ABC


# Letter labels for 10 options
_OPTION_LETTERS = list(string.ascii_uppercase[:10])  # A-J

# Fixed seed for deterministic shuffling within each category
_SHUFFLE_SEED = 888


class MmluProDataset(ABC):
    def __init__(
        self,
        split: Union[Literal['dev'], Literal['train'], Literal['val'], Literal['test']],
        stratified_limit: int = 0,
    ) -> None:
        self._split = split
        self._data = self._load_split(split, stratified_limit)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu_pro'

    @staticmethod
    def _load_split(split: str, stratified_limit: int = 0) -> list:
        from datasets import load_dataset

        # val/dev -> HF validation split (70 items)
        if split in ("val", "dev"):
            ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
            records = list(ds)
            rng = np.random.default_rng(_SHUFFLE_SEED)
            indices = rng.permutation(len(records))
            records = [records[i] for i in indices]
            if stratified_limit > 0 and stratified_limit < len(records):
                records = records[:stratified_limit]
            print(f"[MMLU-Pro] Loaded {len(records)} validation items for split='{split}'")
            return records

        # train and test both come from HF 'test' split (12032 items)
        # They are non-overlapping stratified subsets.
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        all_records = list(ds)

        # Group by category and shuffle within each category deterministically
        category_data: Dict[str, list] = {}
        for rec in all_records:
            cat = rec.get("category", "unknown")
            category_data.setdefault(cat, []).append(rec)

        rng = np.random.default_rng(_SHUFFLE_SEED)
        categories_sorted = sorted(category_data.keys())
        for cat in categories_sorted:
            indices = rng.permutation(len(category_data[cat]))
            category_data[cat] = [category_data[cat][i] for i in indices]

        total_count = len(all_records)

        if split == "train":
            # Take first N items per category (proportional to category size)
            return _stratified_slice(
                category_data, categories_sorted, total_count,
                limit=stratified_limit, offset_limit=0,
            )
        else:
            # split == "test": take next N items per category, after skipping
            # the train portion. This ensures zero overlap.
            # Train always uses 500 items (from dataset_config.json).
            train_limit = 500
            return _stratified_slice(
                category_data, categories_sorted, total_count,
                limit=stratified_limit, offset_limit=train_limit,
            )

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> dict:
        return self._data[index]

    @staticmethod
    def record_to_input(record: dict) -> Dict[str, Any]:
        question = record["question"]
        options = record["options"]

        lines = [question]
        for i, opt in enumerate(options):
            lines.append(f"Option {_OPTION_LETTERS[i]}: {opt}")

        demo_question = "\n".join(lines)
        return {"task": demo_question}

    @staticmethod
    def record_to_target_answer(record: dict) -> str:
        """Return the letter answer (A-J)."""
        answer = record.get("answer", "")
        if isinstance(answer, str) and len(answer) == 1 and answer.upper() in _OPTION_LETTERS:
            return answer.upper()
        # Fallback: use answer_index
        answer_index = record.get("answer_index")
        if answer_index is not None and 0 <= answer_index <= 9:
            return _OPTION_LETTERS[answer_index]
        return str(answer)

    @staticmethod
    def postprocess_answer(answer: Union[str, List[str]]) -> str:
        """Extract letter answer (A-J) from model response."""
        import re
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if not answer:
            return ""

        valid_letters = set(_OPTION_LETTERS)

        # Look for "answer is X" pattern
        ans_pos = answer.find("answer is")
        if ans_pos != -1:
            after = answer[ans_pos + len("answer is"):].strip(":").strip()
            if after and after[0].upper() in valid_letters:
                return after[0].upper()

        # Look for "Option X" or "(X)" patterns
        option_match = re.search(r'(?:Option|option)\s+([A-J])', answer)
        if option_match:
            return option_match.group(1).upper()
        paren_match = re.search(r'\(([A-J])\)', answer)
        if paren_match:
            return paren_match.group(1).upper()

        # Try first character — only if response is very short (e.g. "A", "B.", "C)")
        stripped = answer.strip().rstrip(".),:;")
        if len(stripped) <= 2 and stripped and stripped[0].upper() in valid_letters:
            return stripped[0].upper()

        # Scan for standalone letter A-J — take the last match
        # (avoids picking up "I" from "I think..." at the start)
        matches = re.findall(r'(?<![a-zA-Z])([A-J])(?![a-zA-Z])', answer)
        if matches:
            return matches[-1]

        return ""


def _compute_per_category_counts(
    categories_sorted: list,
    category_sizes: Dict[str, int],
    total_count: int,
    limit: int,
) -> Dict[str, int]:
    """Compute how many items to take from each category (proportional + remainder)."""
    counts: Dict[str, int] = {}
    remaining = limit
    for i, cat in enumerate(categories_sorted):
        if i == len(categories_sorted) - 1:
            n = remaining
        else:
            n = int(category_sizes[cat] / total_count * limit)
        n = min(n, category_sizes[cat])
        n = max(n, 0)
        counts[cat] = n
        remaining -= n
    return counts


def _stratified_slice(
    category_data: Dict[str, list],
    categories_sorted: list,
    total_count: int,
    limit: int,
    offset_limit: int,
) -> list:
    """Take a stratified slice from shuffled category data.

    Args:
        offset_limit: If > 0, skip this many items per category (using the same
            proportional formula) before sampling. This ensures train (offset=0)
            and test (offset=train_limit) are non-overlapping.
    """
    category_sizes = {cat: len(category_data[cat]) for cat in categories_sorted}

    # Compute per-category offset (how many to skip)
    if offset_limit > 0:
        offset_counts = _compute_per_category_counts(
            categories_sorted, category_sizes, total_count, offset_limit,
        )
    else:
        offset_counts = {cat: 0 for cat in categories_sorted}

    if limit <= 0:
        result = []
        for cat in categories_sorted:
            off = offset_counts[cat]
            result.extend(category_data[cat][off:])
        print(f"[MMLU-Pro] Loaded {len(result)} items (no limit, offset={offset_limit})")
        return result

    # Compute per-category sample counts
    sample_counts = _compute_per_category_counts(
        categories_sorted, category_sizes, total_count, limit,
    )

    print(f"[MMLU-Pro] Stratified sampling: {limit} items (offset={offset_limit})")
    print(f"  Categories: {len(categories_sorted)}")

    sampled = []
    for cat in categories_sorted:
        off = offset_counts[cat]
        n = sample_counts[cat]
        available = len(category_data[cat]) - off
        n = min(n, available)
        sampled.extend(category_data[cat][off : off + n])
        print(f"  {cat}: {n} items (offset={off}, available={available})")

    print(f"  Total: {len(sampled)} items")
    return sampled
