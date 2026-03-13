from collections import Counter
import json

from datasets import load_dataset
import matplotlib.pyplot as plt


def decode_batch(batch):
    if "text" in batch:
        batch["text"] = [json.loads(t) if isinstance(t, str) else t for t in batch["text"]]
    return batch


# DATA = ["ecg-qa-ptbxl-250-2500", "ecg-instruct-45k-250-2500"]
# DATA = ["ecg-comprehension-bpm-float-100000-250-2500", "ecg-comprehension-bpm-int-100000-250-2500", "ecg-comprehension-r-peak-count-100000-250-2500"]
# DATA = ["ecg-comprehension-r-peak-count-789480-20000-250-2500", "ecg-comprehension-r-peak-count-789480-5000-250-2500", "ecg-comprehension-r-peak-count-789480-10000-250-2500"]
DATA = ["ecg-r1-no-rl", "ecg-qa-cot-not-cot"]
SPLITS = ["train", "test"]

# Choose one or both: ["answers"], ["questions"], ["answers", "questions"]
PLOT_TARGETS = ["answers", "questions"]

MOST_COMMON = 40
MAX_TEXT_CHARS = 40

COLOR_MAP = {
    "answers": "#4F46E5",   # indigo
    "questions": "#0EA5A4",  # teal
}


def truncate_text(text, max_chars):
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}…"


def get_turn_key(data_name, split):
    return "content" if data_name == "ecg-r1-no-rl" and split == "train" else "value"


def collect_turn_texts(dataset, turn_key, target):
    if target == "answers":
        start = 1
    elif target == "questions":
        start = 0
    else:
        raise ValueError(f"Unsupported target: {target}")

    values = []
    for item in dataset:
        for turn in item["text"][start::2]:
            text = turn.get(turn_key, "") if isinstance(turn, dict) else ""
            if text:
                values.append(text.replace("\n", "").replace("<ecg>", "").replace("<image>", "").strip())
    return values


def plot_top_counts(counter, target, data_name, split, top_k, max_text_chars):
    top_items = counter.most_common(top_k)
    if not top_items:
        print(f"No {target} found for {data_name} ({split}); skipping.")
        return

    labels, counts = zip(*reversed(top_items))
    labels = [truncate_text(label, max_text_chars) for label in labels]

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#FBFBFC")
    ax.set_facecolor("#FBFBFC")

    bars = ax.barh(
        range(len(labels)),
        counts,
        color=COLOR_MAP[target],
        edgecolor="none",
        height=0.68,
    )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, color="#1F2937")
    ax.tick_params(axis="x", labelsize=10, colors="#4B5563")
    ax.tick_params(axis="y", length=0)

    ax.set_xlabel("Count", fontsize=11, color="#111827", labelpad=8)
    ax.set_title(
        f"Top {top_k} most common {target} — {data_name} ({split})",
        fontsize=13,
        fontweight="semibold",
        color="#111827",
        pad=12,
    )

    ax.xaxis.grid(True, linestyle="-", linewidth=0.8, alpha=0.18, color="#64748B")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")

    ax.bar_label(bars, padding=5, fontsize=9, color="#334155", fmt="%d")

    fig.tight_layout()
    output = f"top{top_k}_{target}_{data_name}_{split}.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")

TITLE_MAPPING = {
    "ecg-qa-cot-not-cot" : "ECG-QA-CoT",
    "ecg-r1-no-rl": "ECG-Instruct"
}

def main():
    for data_name in DATA:
        for split in SPLITS:
            turn_key = get_turn_key(data_name, split)
            dataset = load_dataset(
                f"willxxy/{data_name}", split=f"fold1_{split}"
            ).with_transform(decode_batch)

            for target in PLOT_TARGETS:
                texts = collect_turn_texts(dataset, turn_key, target)
                plot_top_counts(
                    Counter(texts),
                    target=target,
                    data_name=TITLE_MAPPING[data_name],
                    split=split,
                    top_k=MOST_COMMON,
                    max_text_chars=MAX_TEXT_CHARS,
                )


if __name__ == "__main__":
    main()
