"""GPTSwarm training — MMLU dataset."""

import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Datasets.mmlu_dataset import MMLUDataset
from Datasets.math_dataset import MATH_get_predict


def _make_loaders(dataset_root):
    def load_train(limit):
        ds = MMLUDataset("dev", data_root=dataset_root, stratified_limit=limit)
        items = []
        for i in range(len(ds)):
            record = ds[i]
            items.append({
                "item_id": i,
                "task": ds.record_to_input(record)["task"],
                "answer": ds.record_to_target_answer(record),
            })
        return items

    def load_val(limit):
        ds = MMLUDataset("val", data_root=dataset_root, stratified_limit=limit if limit > 0 else 125)
        items = []
        for i in range(len(ds)):
            record = ds[i]
            items.append({
                "item_id": i,
                "task": ds.record_to_input(record)["task"],
                "answer": ds.record_to_target_answer(record),
            })
        return items

    return load_train, load_val


def evaluate(result_text, item):
    true_answer = item["answer"]
    predict = MATH_get_predict(result_text)
    if predict:
        predict = predict[0].upper()
    else:
        predict = ""
    is_correct = str(predict).strip() == str(true_answer).strip()
    return is_correct, f"pred={predict} gold={true_answer}"


def get_query(item):
    return item["task"]


def get_item_id(item):
    return str(item.get("item_id", ""))


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--dataset-root", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    load_train, load_val = _make_loaders(pre_args.dataset_root)

    from MAR.GPTSwarm.training import main
    main(
        default_dataset="mmlu",
        load_train_fn=load_train,
        load_val_fn=load_val,
        evaluate_fn=evaluate,
        get_query_fn=get_query,
        get_item_id_fn=get_item_id,
    )
