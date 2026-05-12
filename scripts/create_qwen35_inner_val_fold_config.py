#!/usr/bin/env python3
"""
Generate fold-specific Qwen3.5-9B QLoRA config for corrected 10-fold CV.

Correct methodology:
- Train on oversampled inner_train
- Monitor/evaluate during training on validation
- Do NOT use test fold in training config
"""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = ROOT / "src" / "integration" / "training_scripts" / "ft"

BASE_DATA_DIR = "data/qwen35_oversampling_10fold_cv/inner_splits"
OUTPUT_BASE_DIR = "outputs/qwen35_9b_qlora_oversampling_10fold_cv_inner_val"


def create_config(fold: int):
    fold_name = f"fold_{fold:02d}"

    train_file = f"{BASE_DATA_DIR}/{fold_name}/train_oversampled.jsonl"
    validation_file = f"{BASE_DATA_DIR}/{fold_name}/validation.jsonl"
    output_dir = f"{OUTPUT_BASE_DIR}/{fold_name}"

    config_text = f"""# Corrected Experiment - {fold_name}
# Pure 10-fold CV + inner validation
# Qwen3.5-9B + QLoRA + oversampling
# Test fold is NOT used during training.
# Training: oversampled inner_train
# Training-time evaluation: validation
# Final classification evaluation: held-out test.jsonl outside this config

model_path: "Qwen/Qwen3.5-9B"
method: "qlora"

training:
  learning_rate: 0.00005
  batch_size: 1
  gradient_accumulation_steps: 8
  num_epochs: 2
  max_steps:
  warmup_steps: 20
  max_grad_norm: 1.0
  weight_decay: 0.01
  fp16: false
  bf16: true
  gradient_checkpointing: true

lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

quantization:
  enabled: true
  type: "nf4"
  compute_dtype: "bfloat16"
  double_quant: true

data:
  dataset_name: "{train_file}"
  dataset_split: "train"
  max_samples:
  max_length: 1024
  num_workers: 0

evaluation:
  eval_dataset: "{validation_file}"
  eval_split: "train"
  eval_steps: 50
  eval_samples: 100
  compute_perplexity: true

checkpointing:
  output_dir: "{output_dir}"
  save_steps: 100
  save_total_limit: 3
  resume_from_checkpoint:

logging:
  logging_steps: 10
  log_level: "INFO"

device:
  device: "cuda"

advanced:
  seed: 42
"""

    config_path = CONFIG_DIR / f"config_qwen35_9b_10fold_cv_inner_val_{fold_name}.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    print("Created config:")
    print(config_path.relative_to(ROOT))
    print()
    print("Train file:")
    print(train_file)
    print("Validation file:")
    print(validation_file)
    print("Output dir:")
    print(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    create_config(args.fold)


if __name__ == "__main__":
    main()
