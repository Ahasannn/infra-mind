"""GPTSwarm training — GSM-Hard dataset."""

import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from Datasets.gsm_hard_dataset import gsm_hard_data_process, gsm_get_predict


def _split_data(seed=1234):
    """Load GSM-Hard and split into train/val/test.

    1319 items total, shuffled deterministically:
      train: [0:500]   = 500 items
      val:   [500:625]  = 125 items
      test:  [625:1125] = 500 items
    """
    raw = load_dataset("reasoning-machines/gsm-hard", split="train")
    all_data = gsm_hard_data_process(raw)
    indices = list(range(len(all_data)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train = [all_data[i] for i in indices[:500]]
    val = [all_data[i] for i in indices[500:625]]
    test = [all_data[i] for i in indices[625:1125]]
    return train, val, test


_train_data, _val_data, _test_data = None, None, None

def _ensure_loaded():
    global _train_data, _val_data, _test_data
    if _train_data is None:
        _train_data, _val_data, _test_data = _split_data()


def load_train(limit):
    _ensure_loaded()
    data = _train_data
    if limit and limit > 0:
        data = data[:limit]
    return data


def load_val(limit):
    _ensure_loaded()
    data = _val_data
    if limit and limit > 0:
        data = data[:limit]
    return data


def evaluate(result_text, item):
    true_answer = item["answer"]
    predict = gsm_get_predict(result_text)
    try:
        is_correct = float(predict) == float(true_answer)
    except (ValueError, TypeError):
        is_correct = False
    return is_correct, f"pred={predict} gold={true_answer}"


def get_query(item):
    return item["task"]


def get_item_id(item):
    return str(item.get("id", ""))


if __name__ == "__main__":
    from MAR.GPTSwarm.training import main
    main(
        default_dataset="gsm_hard",
        load_train_fn=load_train,
        load_val_fn=load_val,
        evaluate_fn=evaluate,
        get_query_fn=get_query,
        get_item_id_fn=get_item_id,
    )
