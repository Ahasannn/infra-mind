from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from MAR.Tools.coding.python_executor import PyExecutor
from MAR.Tools.reader.readers import JSONLReader

from Datasets.mbpp_dataset import MbppDataset
from Datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict
from Datasets.gsm_hard_dataset import gsm_hard_data_process, gsm_get_predict as gsm_hard_get_predict
from Datasets.math_dataset import MATH_get_predict, MATH_is_correct, load_math_dataset
from Datasets.mmlu_dataset import MMLUDataset
from Datasets.mmlu_pro_dataset import MmluProDataset


@dataclass(frozen=True)
class InfraMindSample:
    query: str
    tests: Optional[List[str]]
    answer: Optional[str]
    item_id: object
    metadata: Dict[str, object] = field(default_factory=dict)


def _resolve_item_id(row: object, fallback: int) -> object:
    keys = ("task_id", "item_id", "id", "ID", "index", "idx")
    getter = row.get if hasattr(row, "get") else None
    for key in keys:
        value = None
        if getter is not None:
            value = getter(key)
        elif isinstance(row, dict):
            value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            try:
                if value != value:
                    continue
            except Exception:
                pass
        if value == "":
            continue
        return value
    return fallback


def _extract_code(response: str) -> str:
    import re

    pattern = r"```python(.*?)```"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _postprocess_mmlu_pro_answer(answer: Union[str, List[str]]) -> str:
    """Extract letter answer (A-J) from model response for MMLU-Pro."""
    import re
    import string as _string
    valid_letters = set(_string.ascii_uppercase[:10])  # A-J

    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    if not isinstance(answer, str):
        raise ValueError("Expected answer as string.")
    if not answer:
        return ""

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

    # Try first character — only if response is very short (e.g. "A", "B.")
    stripped = answer.strip().rstrip(".),:;")
    if len(stripped) <= 2 and stripped and stripped[0].upper() in valid_letters:
        return stripped[0].upper()

    # Scan for standalone letter A-J — take the last match
    matches = re.findall(r'(?<![a-zA-Z])([A-J])(?![a-zA-Z])', answer)
    if matches:
        return matches[-1]

    return ""


def _postprocess_mmlu_answer(answer: Union[str, List[str]]) -> str:
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    if not isinstance(answer, str):
        raise ValueError("Expected answer as string.")
    if answer:
        ans_pos = answer.find("answer is")
        if ans_pos != -1:
            answer = answer[ans_pos + len("answer is") :].strip(":").strip().strip("Option").strip()
        answer = answer[0]
    return answer


class InfraMindDataset(ABC):
    dataset_name: str
    role_domain: str
    prompt_file: str

    def __init__(self, split: str, seed: int = 42) -> None:
        self.split = split
        self.seed = seed
        self._samples = self._load_samples()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> InfraMindSample:
        return self._samples[index]

    def sample(self, *, limit: int = 0, shuffle: bool = True, seed: Optional[int] = None) -> List[InfraMindSample]:
        indices = list(range(len(self._samples)))
        if shuffle:
            rng = random.Random(self.seed if seed is None else seed)
            rng.shuffle(indices)
        if limit and limit > 0:
            indices = indices[: int(limit)]
        return [self._samples[i] for i in indices]

    @abstractmethod
    def _load_samples(self) -> List[InfraMindSample]:
        raise NotImplementedError

    @abstractmethod
    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[InfraMindSample],
    ) -> Tuple[float, Dict[str, object]]:
        raise NotImplementedError


class MbppAdapter(InfraMindDataset):
    dataset_name = "mbpp"
    role_domain = "Code"
    prompt_file = os.path.join("MAR", "Roles", "FinalNode", "mbpp.json")

    def _load_samples(self) -> List[InfraMindSample]:
        ds = MbppDataset(self.split)
        samples: List[InfraMindSample] = []
        for idx, row in ds.df.iterrows():
            query = str(row.get("task") or row["text"])
            tests = list(row["test_list"])
            item_id = _resolve_item_id(row, idx)
            samples.append(InfraMindSample(query=query, tests=tests, answer=None, item_id=item_id))
        return samples

    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[InfraMindSample],
    ) -> Tuple[float, Dict[str, object]]:
        if not tests:
            return 0.0, {"is_solved": False, "feedback": "Missing tests."}
        code = _extract_code(response)
        is_solved, feedback, _ = PyExecutor().execute(code, list(tests), timeout=30, verbose=False)
        return float(1.0 if is_solved else 0.0), {"is_solved": bool(is_solved), "feedback": feedback}


class HumanEvalAdapter(InfraMindDataset):
    dataset_name = "humaneval"
    role_domain = "Code"
    prompt_file = os.path.join("MAR", "Roles", "FinalNode", "humaneval.json")

    def __init__(
        self,
        split: str,
        *,
        dataset_path: Optional[str] = None,
        split_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.dataset_path = dataset_path or os.path.join("Datasets", "humaneval", "humaneval-py.jsonl")
        self.split_ratio = split_ratio
        super().__init__(split, seed=seed)

    def _load_samples(self) -> List[InfraMindSample]:
        # Use HumanEvalDataset (pandas-based) for identical splits across all methods
        from Datasets.humaneval_dataset import HumanEvalDataset
        dataset = HumanEvalDataset(self.split, split_ratio=self.split_ratio, seed=self.seed)
        samples: List[InfraMindSample] = []
        for idx in range(len(dataset)):
            row = dataset[idx]
            query = str(row.get("prompt", ""))
            test = row.get("test", "")
            item_id = _resolve_item_id(row, idx)
            tests = [str(test)] if test else []
            samples.append(InfraMindSample(query=query, tests=tests, answer=None, item_id=item_id))
        return samples

    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[InfraMindSample],
    ) -> Tuple[float, Dict[str, object]]:
        if not tests:
            return 0.0, {"is_solved": False, "feedback": "Missing tests."}
        code = _extract_code(response)
        is_solved, feedback, _ = PyExecutor().execute(code, list(tests), timeout=30, verbose=False)
        return float(1.0 if is_solved else 0.0), {"is_solved": bool(is_solved), "feedback": feedback}


class Gsm8kAdapter(InfraMindDataset):
    dataset_name = "gsm8k"
    role_domain = "Math"
    prompt_file = os.path.join("MAR", "Roles", "FinalNode", "gsm8k.json")

    def __init__(
        self,
        split: str,
        *,
        dataset_path: Optional[str] = None,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        split_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.dataset_path = dataset_path or os.path.join("Datasets", "gsm8k", "gsm8k.jsonl")
        self.train_path = train_path
        self.test_path = test_path
        self.split_ratio = split_ratio
        super().__init__(split, seed=seed)

    def _load_samples(self) -> List[InfraMindSample]:
        reader = JSONLReader()
        if self.split in ("train", "test") and (self.train_path or self.test_path):
            path = self.train_path if self.split == "train" else self.test_path
            if not path:
                path = self.dataset_path
            records = reader.parse_file(path)
            processed = gsm_data_process(records)
        else:
            records = reader.parse_file(self.dataset_path)
            processed = gsm_data_process(records)
            indices = list(range(len(processed)))
            rng = random.Random(self.seed)
            rng.shuffle(indices)
            split_index = int(len(indices) * self.split_ratio)
            train_indices = indices[:split_index]
            test_indices = indices[split_index:]
            if self.split == "train":
                processed = [processed[i] for i in train_indices]
            elif self.split == "test":
                processed = [processed[i] for i in test_indices]

        samples: List[InfraMindSample] = []
        for idx, row in enumerate(processed):
            query = str(row.get("task", ""))
            answer = str(row.get("answer", "")).strip()
            item_id = _resolve_item_id(row, idx)
            samples.append(InfraMindSample(query=query, tests=None, answer=answer, item_id=item_id))
        return samples

    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[InfraMindSample],
    ) -> Tuple[float, Dict[str, object]]:
        answer = sample.answer if sample else ""
        pred = gsm_get_predict(response)
        pred_value = _safe_float(str(pred))
        gold_value = _safe_float(str(answer))
        if pred_value is not None and gold_value is not None:
            is_solved = pred_value == gold_value
        else:
            is_solved = str(pred).strip() == str(answer).strip()
        return float(1.0 if is_solved else 0.0), {"is_solved": bool(is_solved), "pred": pred, "gold": answer}


class GsmHardAdapter(InfraMindDataset):
    dataset_name = "gsm_hard"
    role_domain = "Math"
    prompt_file = os.path.join("MAR", "Roles", "FinalNode", "gsm8k.json")

    def _load_samples(self) -> List[InfraMindSample]:
        from datasets import load_dataset as hf_load_dataset

        raw = hf_load_dataset("reasoning-machines/gsm-hard", split="train")
        processed = gsm_hard_data_process(raw)

        # Fixed splits from 1319 shuffled items (seed=1234 to match baselines):
        #   train: [0:500], val: [500:625], test: [625:1125]
        indices = list(range(len(processed)))
        rng = random.Random(1234)
        rng.shuffle(indices)

        if self.split == "train":
            active_indices = indices[:500]
        elif self.split in ("val", "dev"):
            active_indices = indices[500:625]
        elif self.split == "test":
            active_indices = indices[625:1125]
        else:
            active_indices = indices

        samples: List[InfraMindSample] = []
        for idx in active_indices:
            row = processed[idx]
            query = str(row.get("task", ""))
            answer = str(row.get("answer", "")).strip()
            item_id = _resolve_item_id(row, idx)
            samples.append(InfraMindSample(query=query, tests=None, answer=answer, item_id=item_id))
        return samples

    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[InfraMindSample],
    ) -> Tuple[float, Dict[str, object]]:
        answer = sample.answer if sample else ""
        pred = gsm_hard_get_predict(response)
        pred_value = _safe_float(str(pred))
        gold_value = _safe_float(str(answer))
        if pred_value is not None and gold_value is not None:
            is_solved = pred_value == gold_value
        else:
            is_solved = str(pred).strip() == str(answer).strip()
        return float(1.0 if is_solved else 0.0), {"is_solved": bool(is_solved), "pred": pred, "gold": answer}


class MathAdapter(InfraMindDataset):
    dataset_name = "math"
    role_domain = "Math"
    prompt_file = os.path.join("MAR", "Roles", "FinalNode", "math.json")

    def __init__(
        self,
        split: str,
        *,
        dataset_root: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.dataset_root = dataset_root or os.path.join("Datasets", "MATH")
        super().__init__(split, seed=seed)

    def _load_samples(self) -> List[InfraMindSample]:
        records = load_math_dataset(self.dataset_root, split=self.split)
        samples: List[InfraMindSample] = []
        for idx, row in enumerate(records):
            query = str(row.get("problem", ""))
            answer = str(row.get("solution", ""))
            item_id = _resolve_item_id(row, idx)
            samples.append(InfraMindSample(query=query, tests=None, answer=answer, item_id=item_id))
        return samples

    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[InfraMindSample],
    ) -> Tuple[float, Dict[str, object]]:
        answer = sample.answer if sample else ""
        pred = MATH_get_predict(response)
        is_solved = bool(MATH_is_correct(pred, answer))
        gold = MATH_get_predict(answer) if answer else ""
        return float(1.0 if is_solved else 0.0), {"is_solved": is_solved, "pred": pred, "gold": gold}


class MmluAdapter(InfraMindDataset):
    dataset_name = "mmlu"
    role_domain = "Commonsense"
    prompt_file = os.path.join("MAR", "Roles", "FinalNode", "mmlu.json")

    def __init__(
        self,
        split: str,
        *,
        dataset_root: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.dataset_root = dataset_root
        super().__init__(split, seed=seed)

    def _resolve_split(self) -> str:
        if self.split in ("train", "dev"):
            return "dev"
        if self.split == "val":
            return "val"
        return "test"

    def _load_samples(self) -> List[InfraMindSample]:
        split = self._resolve_split()
        dataset = MMLUDataset(split, data_root=self.dataset_root)
        samples: List[InfraMindSample] = []
        for idx in range(len(dataset)):
            record = dataset[idx]
            query = dataset.record_to_input(record)["task"]
            answer = dataset.record_to_target_answer(record)
            item_id = _resolve_item_id(record, idx)
            samples.append(InfraMindSample(query=query, tests=None, answer=answer, item_id=item_id))
        return samples

    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[InfraMindSample],
    ) -> Tuple[float, Dict[str, object]]:
        answer = sample.answer if sample else ""
        pred = _postprocess_mmlu_answer(response)
        is_solved = str(pred).strip().upper() == str(answer).strip().upper()
        return float(1.0 if is_solved else 0.0), {"is_solved": is_solved, "pred": pred, "gold": answer}


class MmluProAdapter(InfraMindDataset):
    dataset_name = "mmlu_pro"
    role_domain = "Commonsense"
    prompt_file = os.path.join("MAR", "Roles", "FinalNode", "mmlu.json")

    def __init__(
        self,
        split: str,
        *,
        seed: int = 42,
    ) -> None:
        super().__init__(split, seed=seed)

    def _load_samples(self) -> List[InfraMindSample]:
        # MmluProDataset handles split mapping internally:
        #   train → stratified from HF test (first N per category)
        #   test  → stratified from HF test (next N, non-overlapping)
        #   val/dev → HF validation (70 items)
        dataset = MmluProDataset(self.split)
        samples: List[InfraMindSample] = []
        for idx in range(len(dataset)):
            record = dataset[idx]
            query = dataset.record_to_input(record)["task"]
            answer = dataset.record_to_target_answer(record)
            item_id = _resolve_item_id(record, idx)
            samples.append(InfraMindSample(query=query, tests=None, answer=answer, item_id=item_id))
        return samples

    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[InfraMindSample],
    ) -> Tuple[float, Dict[str, object]]:
        answer = sample.answer if sample else ""
        pred = _postprocess_mmlu_pro_answer(response)
        is_solved = str(pred).strip().upper() == str(answer).strip().upper()
        return float(1.0 if is_solved else 0.0), {"is_solved": is_solved, "pred": pred, "gold": answer}


def get_dataset_adapter(
    dataset_name: str,
    *,
    split: str,
    seed: int = 42,
    dataset_path: Optional[str] = None,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    dataset_root: Optional[str] = None,
    split_ratio: float = 0.2,
) -> InfraMindDataset:
    name = dataset_name.strip().lower()
    if name == "mbpp":
        return MbppAdapter(split, seed=seed)
    if name == "humaneval":
        return HumanEvalAdapter(
            split,
            dataset_path=dataset_path,
            split_ratio=split_ratio,
            seed=seed,
        )
    if name == "gsm8k":
        return Gsm8kAdapter(
            split,
            dataset_path=dataset_path,
            train_path=train_path,
            test_path=test_path,
            split_ratio=split_ratio,
            seed=seed,
        )
    if name == "gsm_hard":
        return GsmHardAdapter(
            split,
            seed=seed,
        )
    if name == "math":
        return MathAdapter(
            split,
            dataset_root=dataset_root or dataset_path,
            seed=seed,
        )
    if name == "mmlu":
        return MmluAdapter(
            split,
            dataset_root=dataset_root or dataset_path,
            seed=seed,
        )
    if name == "mmlu_pro":
        return MmluProAdapter(
            split,
            seed=seed,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def available_datasets() -> Sequence[str]:
    return ("mbpp", "humaneval", "gsm8k", "gsm_hard", "math", "mmlu", "mmlu_pro")
