#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


def replace_yaml_value(text, key, value):
    pattern = rf'^(\s*{re.escape(key)}:\s*).*$'
    replacement = rf'\1"{value}"'
    return re.sub(pattern, replacement, text, flags=re.MULTILINE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--train_size", required=True, choices=["100", "200", "300", "full"])
    args = parser.parse_args()

    fold_name = f"fold_{args.fold:02d}"
    size_name = f"size_{args.train_size}"

    template_candidates = [
        Path(f"src/integration/training_scripts/ft/config_qwen35_9b_10fold_cv_inner_val_undersampling_{fold_name}.yaml"),
        Path("src/integration/training_scripts/ft/config_qwen35_9b_10fold_cv_inner_val_undersampling_fold_01.yaml"),
        Path(f"src/integration/training_scripts/ft/config_qwen35_9b_10fold_cv_inner_val_fold_{args.fold:02d}.yaml"),
        Path("src/integration/training_scripts/ft/config_qwen35_9b_10fold_cv_inner_val_fold_01.yaml"),
    ]

    template_path = None
    for p in template_candidates:
        if p.exists():
            template_path = p
            break

    if template_path is None:
        raise FileNotFoundError("No suitable config template found.")

    if args.train_size == "full":
        train_file = f"data/qwen35_oversampling_10fold_cv/inner_splits/{fold_name}/train_undersampled.jsonl"
    else:
        train_file = f"data/qwen35_oversampling_10fold_cv/inner_splits/{fold_name}/train_size_{args.train_size}_undersampled.jsonl"

    validation_file = f"data/qwen35_oversampling_10fold_cv/inner_splits/{fold_name}/validation.jsonl"

    output_dir = (
        f"outputs/qwen35_9b_qlora_train_size_undersampling_10fold_cv_inner_val_v2/"
        f"{size_name}/{fold_name}"
    )

    if not Path(train_file).exists():
        raise FileNotFoundError(f"Missing train file: {train_file}")

    if not Path(validation_file).exists():
        raise FileNotFoundError(f"Missing validation file: {validation_file}")

    text = template_path.read_text(encoding="utf-8")

    text = replace_yaml_value(text, "dataset_name", train_file)
    text = replace_yaml_value(text, "eval_dataset", validation_file)
    text = replace_yaml_value(text, "output_dir", output_dir)

    note = (
        "# Qwen3.5-9B QLoRA train-size undersampling experiment V2\n"
        "# Train sizes 100/200/300 use balanced undersampled subsets.\n"
        "# Full uses train_undersampled.jsonl.\n"
        "# Test fold is used only after training by strict_v2 evaluator.\n\n"
    )

    text = re.sub(r'^(#.*\n)*', note, text, count=1)

    out_path = Path(
        f"src/integration/training_scripts/ft/"
        f"config_qwen35_9b_train_size_undersampling_v2_{size_name}_{fold_name}.yaml"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    print("Created config:")
    print(out_path)
    print()
    print("Template used:")
    print(template_path)
    print()
    print("Train file:")
    print(train_file)
    print("Validation file:")
    print(validation_file)
    print("Output dir:")
    print(output_dir)


if __name__ == "__main__":
    main()
