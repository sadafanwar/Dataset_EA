#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def label_of(row):
    label = str(row.get("output", "")).strip()
    if label not in {"trace", "no_trace"}:
        raise ValueError(f"Invalid label: {label}")
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--train_size", required=True, choices=["100", "200", "300", "500", "full"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fold_name = f"fold_{args.fold:02d}"
    inner_dir = Path("data/qwen35_oversampling_10fold_cv/inner_splits") / fold_name

    inner_train_path = inner_dir / "inner_train.jsonl"
    full_undersampled_path = inner_dir / "train_undersampled.jsonl"

    inner_train = read_jsonl(inner_train_path)

    if args.train_size == "full":
        if not full_undersampled_path.exists():
            raise FileNotFoundError(f"Missing: {full_undersampled_path}")
        subset = read_jsonl(full_undersampled_path)
        out_path = full_undersampled_path
    else:
        n = int(args.train_size)
        per_class = n // 2

        grouped = defaultdict(list)
        for r in inner_train:
            grouped[label_of(r)].append(r)

        rng = random.Random(args.seed + args.fold + n)

        trace_rows = rng.sample(grouped["trace"], per_class)
        no_trace_rows = rng.sample(grouped["no_trace"], per_class)

        subset = trace_rows + no_trace_rows
        rng.shuffle(subset)

        out_path = inner_dir / f"train_size_{n}_undersampled.jsonl"
        write_jsonl(out_path, subset)

    counts = Counter(label_of(r) for r in subset)

    print(f"Fold: {fold_name}")
    print(f"Train size: {args.train_size}")
    print(f"Output: {out_path}")
    print(f"Total: {len(subset)}")
    print(f"Counts: {counts}")

    if counts["trace"] != counts["no_trace"]:
        raise RuntimeError(f"Subset not balanced: {counts}")


if __name__ == "__main__":
    main()
