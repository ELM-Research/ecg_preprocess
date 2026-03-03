import argparse
import json
import os

import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from huggingface_hub import login
from sklearn.model_selection import StratifiedShuffleSplit


def encode_row(item: dict) -> dict:
    if "text" in item:
        item = dict(item)
        item["text"] = json.dumps(item["text"], ensure_ascii=False, separators=(",", ":"))
    return item


def decode_batch(batch: dict) -> dict:
    if "text" in batch:
        out = []
        for t in batch["text"]:
            try:
                out.append(json.loads(t))
            except Exception:
                out.append(t)
        batch["text"] = out
    return batch


def numeric_answer(item: dict) -> str:
    return str(item["text"][1]["value"])


def stratified_split(data: list[dict], train_ratio: float, seed: int):
    labels = [numeric_answer(item) for item in data]
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(data)), labels))
    train = [data[i] for i in train_idx]
    test = [data[i] for i in test_idx]
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_json", type=str, required=True, help="The mapping json")
    ap.add_argument("--fold_json", type=str, required=True, help="Outputted fold json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo_id", type=str, required=True)
    ap.add_argument("--load", action="store_true")
    args = ap.parse_args()

    if args.load:
        ds = load_dataset(args.repo_id, split="fold1_train").with_transform(decode_batch)
        print(f"fold1_train: {len(ds)} samples")
        print(f"  keys: {list(ds[0].keys())}")
        print(f"  text: {ds[0]['text']}")
        return

    with open(args.map_json) as f:
        data = json.load(f)

    print(f"Dataset Length: {len(data)}")

    folds_dict = {}
    for k in range(args.folds):
        train, test = stratified_split(data, train_ratio=args.train_ratio, seed=args.seed + k)
        folds_dict[f"fold{k + 1}"] = {"train": train, "test": test}
        print(f"Fold {k + 1}: train={len(train)}, test={len(test)}")

    with open(args.fold_json, "w") as f:
        json.dump(folds_dict, f, indent=2)

    print(f"\nSaved {args.folds} folds to {args.fold_json}")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("export HF_TOKEN=hf_xxx and retry")

    login(token=token, new_session=False)

    features = Features({
        "ecg_path": Value("string"),
        "text": Value("string"),
        "name": Value("string"),
    })

    splits = {}
    for fold_name, parts in folds_dict.items():
        splits[f"{fold_name}_train"] = Dataset.from_list([encode_row(d) for d in parts["train"]], features=features)
        splits[f"{fold_name}_test"] = Dataset.from_list([encode_row(d) for d in parts["test"]], features=features)

    DatasetDict(splits).push_to_hub(args.repo_id, token=token)


if __name__ == "__main__":
    main()
