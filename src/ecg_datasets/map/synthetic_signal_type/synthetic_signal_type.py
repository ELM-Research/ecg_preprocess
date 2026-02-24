import argparse
import glob
import json
import os
import string
from collections import defaultdict

import numpy as np
from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import login

from configs.constants import DATA_DIR
from ecg_datasets.map.map_dataset import SyntheticDataset


def build_qa_pair(signal_types: list[str]) -> tuple[str, dict[str, str]]:
    labels = string.ascii_uppercase[: len(signal_types)]
    mapping = dict(zip(signal_types, labels))
    lines = [f"{lbl}) {sig}" for sig, lbl in mapping.items()]
    question = "What type of signal is provided?\n" + "\n".join(lines)
    return question, mapping


def build_instances(
    signal_types: list[str],
    per_type: int,
    segment_len: int,
) -> list[dict]:
    question, mapping = build_qa_pair(signal_types)
    ecg_paths = glob.glob(f"{DATA_DIR}/mimic_iv/preprocessed_{segment_len}/*.npy")
    if not ecg_paths:
        raise FileNotFoundError(
            f"No .npy files in {DATA_DIR}/mimic_iv/preprocessed_{segment_len}/"
        )
    ecg_paths.sort()

    data = []
    for signal_type in signal_types:
        for i in range(per_type):
            ecg_path = ecg_paths[i % len(ecg_paths)] if signal_type == "ecg" else signal_type
            data.append(
                {
                    "ecg_path": ecg_path,
                    "text": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": mapping[signal_type]},
                    ],
                    "name": signal_type,
                }
            )
    return data


def stratified_split(
    data: list[dict], train_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    by_type = defaultdict(list)
    for item in data:
        by_type[item["name"]].append(item)

    train, test = [], []
    for items in by_type.values():
        arr = list(items)
        rng.shuffle(arr)
        n_train = int(round(len(arr) * train_ratio))
        train.extend(arr[:n_train])
        test.extend(arr[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def encode_row(item: dict) -> dict:
    item = dict(item)
    item["text"] = json.dumps(item["text"], ensure_ascii=False, separators=(",", ":"))
    return item


class SyntheticSignalType(SyntheticDataset):
    """Pipeline-compatible wrapper: generates the JSON via main.py --map synthetic_signal_type."""

    def __init__(self, args, logger):
        super().__init__(args, logger)
        dataset_name = "-".join(args.signal_types) + f"-{args.segment_len}"
        self.save_dir_json = f"src/ecg_datasets/map/synthetic_signal_type/{dataset_name}.json"

    def get_map_data(self):
        return build_instances(
            self.args.signal_types, self.args.per_type, self.args.segment_len
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--signal_types",
        nargs="+",
        required=True,
        help="Signal types, e.g. ecg noise flatline",
    )
    ap.add_argument("--per_type", type=int, required=True, help="Instances per signal type")
    ap.add_argument("--segment_len", type=int, default=2500)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo_id", type=str, required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    dataset_name = "-".join(args.signal_types) + f"-{args.segment_len}"
    json_path = f"src/ecg_datasets/map/synthetic_signal_type/{dataset_name}.json"

    data = build_instances(args.signal_types, args.per_type, args.segment_len)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Created {len(data)} instances -> {json_path}")

    features = Features(
        {"ecg_path": Value("string"), "text": Value("string"), "name": Value("string")}
    )

    splits = {}
    for k in range(args.folds):
        train, test = stratified_split(data, args.train_ratio, seed=args.seed + k)

        train_counts = defaultdict(int)
        test_counts = defaultdict(int)
        for d in train:
            train_counts[d["name"]] += 1
        for d in test:
            test_counts[d["name"]] += 1

        print(f"Fold {k+1}: train={dict(train_counts)}, test={dict(test_counts)}")
        splits[f"fold{k+1}_train"] = Dataset.from_list(
            [encode_row(d) for d in train], features=features
        )
        splits[f"fold{k+1}_test"] = Dataset.from_list(
            [encode_row(d) for d in test], features=features
        )

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("export HF_TOKEN=hf_xxx and retry")
    login(token=token, new_session=False)
    DatasetDict(splits).push_to_hub(args.repo_id, token=token)
    print(f"Pushed to {args.repo_id}")


if __name__ == "__main__":
    main()
