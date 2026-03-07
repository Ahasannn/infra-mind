"""GPTSwarm training — MMLU-Pro dataset."""

import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Datasets.mmlu_pro_dataset import MmluProDataset


def _make_loaders():
    def load_train(limit):
        ds = MmluProDataset("train", stratified_limit=limit)
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
        ds = MmluProDataset("val", stratified_limit=limit if limit > 0 else 70)
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


def _postprocess_answer(text):
    """Extract letter answer (A-J) from model response."""
    import re
    import string
    valid_letters = set(string.ascii_uppercase[:10])

    if not text:
        return ""
    ans_pos = text.find("answer is")
    if ans_pos != -1:
        after = text[ans_pos + len("answer is"):].strip(":").strip()
        if after and after[0].upper() in valid_letters:
            return after[0].upper()
    option_match = re.search(r'(?:Option|option)\s+([A-J])', text)
    if option_match:
        return option_match.group(1).upper()
    paren_match = re.search(r'\(([A-J])\)', text)
    if paren_match:
        return paren_match.group(1).upper()
    stripped = text.strip().rstrip(".),:;")
    if len(stripped) <= 2 and stripped and stripped[0].upper() in valid_letters:
        return stripped[0].upper()
    matches = re.findall(r'(?<![a-zA-Z])([A-J])(?![a-zA-Z])', text)
    if matches:
        return matches[-1]
    return ""


def evaluate(result_text, item):
    true_answer = item["answer"]
    predict = _postprocess_answer(result_text)
    is_correct = str(predict).strip().upper() == str(true_answer).strip().upper()
    return is_correct, f"pred={predict} gold={true_answer}"


def get_query(item):
    return item["task"]


def get_item_id(item):
    return str(item.get("item_id", ""))


if __name__ == "__main__":
    load_train, load_val = _make_loaders()

    from MAR.GPTSwarm.training import main
    main(
        default_dataset="mmlu_pro",
        load_train_fn=load_train,
        load_val_fn=load_val,
        evaluate_fn=evaluate,
        get_query_fn=get_query,
        get_item_id_fn=get_item_id,
    )
