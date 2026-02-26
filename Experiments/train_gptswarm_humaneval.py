"""GPTSwarm training — HumanEval dataset."""

import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MAR.Tools.coding.python_executor import PyExecutor
from Datasets.humaneval_dataset import HumanEvalDataset


_CODE_PATTERN = re.compile(r"```python.*?```", re.DOTALL | re.MULTILINE)


def load_train(limit):
    ds = HumanEvalDataset("train", limit=limit)
    return ds.df.to_dict("records")


def load_val(limit):
    # HumanEval is tiny — use a small val split
    ds = HumanEvalDataset("test", limit=limit if limit > 0 else 10)
    return ds.df.to_dict("records")


def evaluate(result_text, item):
    test = item["test"]
    match = _CODE_PATTERN.search(result_text)
    if match:
        code = match.group(0).lstrip("```python\n").rstrip("\n```")
        is_solved, feedback, _ = PyExecutor().execute(code, [test], timeout=100)
        return bool(is_solved), feedback
    return False, "No python code block found."


def get_query(item):
    return item["prompt"]


def get_item_id(item):
    return str(item.get("task_id", ""))


if __name__ == "__main__":
    from MAR.GPTSwarm.training import main
    main(
        default_dataset="humaneval",
        load_train_fn=load_train,
        load_val_fn=load_val,
        evaluate_fn=evaluate,
        get_query_fn=get_query,
        get_item_id_fn=get_item_id,
    )
