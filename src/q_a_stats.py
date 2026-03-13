from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
import json

def decode_batch(batch):
    if "text" in batch:
        batch["text"] = [json.loads(t) if isinstance(t, str) else t for t in batch["text"]]
    return batch

# DATA = ["ecg-qa-ptbxl-250-2500", "ecg-instruct-45k-250-2500"]
# DATA = ["ecg-comprehension-bpm-float-100000-250-2500", "ecg-comprehension-bpm-int-100000-250-2500", "ecg-comprehension-r-peak-count-100000-250-2500"]
# DATA = ["ecg-comprehension-r-peak-count-789480-20000-250-2500", "ecg-comprehension-r-peak-count-789480-5000-250-2500", "ecg-comprehension-r-peak-count-789480-10000-250-2500"]
DATA = ["ecg-r1-no-rl", "ecg-qa-cot-not-cot"]
MOST_COMMON = 40
for data_name in DATA:
    for split in ["train", "test"]:
        if data_name == "ecg-r1-no-rl" and split == "train":
            turn_ind = "content"
        else:
            turn_ind = "value"
        data = load_dataset(f"willxxy/{data_name}", split=f"fold1_{split}").with_transform(decode_batch)
        all_answers = []
        for item in data:
            for turn in item["text"][1::2]:
                all_answers.append(turn[turn_ind])
        answers = Counter(all_answers)
        top20 = answers.most_common(MOST_COMMON)

        labels, counts = zip(*reversed(top20))

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(range(len(labels)), counts, color="#2563eb", edgecolor="none")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Count")
        ax.set_title(f"Top {MOST_COMMON} Most Common Answers for {data_name}; Split: {split}", fontweight="bold", fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.bar_label(bars, padding=4, fontsize=12, color="#555")
        fig.tight_layout()
        fig.savefig(f"top{MOST_COMMON}_answers_{data_name}_{split}.png", dpi=180, bbox_inches="tight")
        plt.close()
