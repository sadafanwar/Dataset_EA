#!/usr/bin/env python3

import argparse
from pathlib import Path
import re


def replace_yaml_value(text, key, value):
    pattern = rf'^(\s*{re.escape(key)}:\s*).*$'
    replacement = rf'\1"{value}"'
    return re.sub(pattern, replacement, text, flags=re.MULTILINE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    fold_name = f"fold_{args.fold:02d}"

    template_candidates = [
        Path(f"src/integration/training_scripts/ft/config_qwen35_9b_10fold_cv_inner_val_fold_{args.fold:02d}.yaml"),
        Path("src/integration/training_scripts/ft/config_qwen35_9b_10fold_cv_inner_val_fold_01.yaml"),
    ]

    template_path = None
    for p in template_candidates:
        if p.exists():
            template_path = p
            break

    if template_path is None:
        raise FileNotFoundError("Could not find an existing inner-validation config template.")

    train_file = f"data/qwen35_oversampling_10fold_cv/inner_splits/{fold_name}/train_undersampled.jsonl"
    validation_file = f"data/qwen35_oversampling_10fold_cv/inner_splits/{fold_name}/validation.jsonl"
    output_dir = f"outputs/qwen35_9b_qlora_undersampling_10fold_cv_inner_val/{fold_name}"

    if not Path(train_file).exists():
        raise FileNotFoundError(f"Missing train undersampled file: {train_file}")

    if not Path(validation_file).exists():
        raise FileNotFoundError(f"Missing validation file: {validation_file}")

    text = template_path.read_text(encoding="utf-8")

    text = replace_yaml_value(text, "dataset_name", train_file)
    text = replace_yaml_value(text, "eval_dataset", validation_file)
    text = replace_yaml_value(text, "output_dir", output_dir)

    note = (
        "# Qwen3.5-9B QLoRA 10-fold CV with INNER VALIDATION and UNDERSAMPLING\n"
        "# Final classification evaluation uses held-out test.jsonl outside this config.\n"
        "# Test fold is not used during training or validation/perplexity monitoring.\n\n"
    )

    text = re.sub(r'^(#.*\n)*', note, text, count=1)

    out_path = Path(
        f"src/integration/training_scripts/ft/"
        f"config_qwen35_9b_10fold_cv_inner_val_undersampling_{fold_name}.yaml"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    print("Created config:")
    print(out_path)
    print("\nTrain file:")
    print(train_file)
    print("Validation file:")
    print(validation_file)
    print("Output dir:")
    print(output_dir)


if __name__ == "__main__":
    main()
