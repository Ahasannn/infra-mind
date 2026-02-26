"""GPTSwarm training — MATH dataset."""

import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Datasets.math_dataset import load_math_dataset, MATH_is_correct, MATH_get_predict


def _make_loaders(dataset_root):
    def load_train(limit):
        return load_math_dataset(dataset_root, "train", stratified_limit=limit)

    def load_val(limit):
        return load_math_dataset(dataset_root, "test", stratified_limit=limit if limit > 0 else 131)

    return load_train, load_val


def evaluate(result_text, item):
    true_answer = item["solution"]
    try:
        predict = MATH_get_predict(result_text)
        is_correct = MATH_is_correct(predict, true_answer)
    except Exception:
        predict = "0"
        is_correct = False
    gold = MATH_get_predict(true_answer) if true_answer else "0"
    return is_correct, f"pred={predict} gold={gold}"


def get_query(item):
    return item["problem"]


def get_item_id(item):
    return str(item.get("id", ""))


if __name__ == "__main__":
    # Pre-parse dataset-root before handing off to training.main()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--dataset-root", type=str, default="Datasets/MATH")
    pre_args, _ = pre_parser.parse_known_args()

    load_train, load_val = _make_loaders(pre_args.dataset_root)

    from MAR.GPTSwarm.training import main
    main(
        default_dataset="math",
        load_train_fn=load_train,
        load_val_fn=load_val,
        evaluate_fn=evaluate,
        get_query_fn=get_query,
        get_item_id_fn=get_item_id,
    )
