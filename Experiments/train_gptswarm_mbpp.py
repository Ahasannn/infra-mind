"""GPTSwarm training — MBPP dataset."""

import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MAR.Tools.coding.python_executor import PyExecutor
from Datasets.mbpp_dataset import MbppDataset


_CODE_PATTERN = re.compile(r"```python.*?```", re.DOTALL | re.MULTILINE)


def load_train(limit):
    ds = MbppDataset("train", limit=limit)
    return ds.df.to_dict("records")


def load_val(limit):
    ds = MbppDataset("val", limit=limit)
    return ds.df.to_dict("records")


def evaluate(result_text, item):
    tests = item["test_list"]
    match = _CODE_PATTERN.search(result_text)
    if match:
        code = match.group(0).lstrip("```python\n").rstrip("\n```")
        is_solved, feedback, _ = PyExecutor().execute(code, list(tests), timeout=100)
        return bool(is_solved), feedback
    return False, "No python code block found."


def get_query(item):
    return item["task"]


def get_item_id(item):
    return str(item.get("task_id", ""))


if __name__ == "__main__":
    from MAR.GPTSwarm.training import main
    main(
        default_dataset="mbpp",
        load_train_fn=load_train,
        load_val_fn=load_val,
        evaluate_fn=evaluate,
        get_query_fn=get_query,
        get_item_id_fn=get_item_id,
    )
