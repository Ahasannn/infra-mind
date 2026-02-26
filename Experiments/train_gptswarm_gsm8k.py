"""GPTSwarm training — GSM8K dataset."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from Datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict


def load_train(limit):
    raw = load_dataset("openai/gsm8k", "main", split="train")
    return gsm_data_process(raw, limit=limit)


def load_val(limit):
    # Use first portion of test as validation
    raw = load_dataset("openai/gsm8k", "main", split="test")
    data = gsm_data_process(raw)
    val_size = limit if limit > 0 else 125
    return data[:val_size]


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
        default_dataset="gsm8k",
        load_train_fn=load_train,
        load_val_fn=load_val,
        evaluate_fn=evaluate,
        get_query_fn=get_query,
        get_item_id_fn=get_item_id,
    )
