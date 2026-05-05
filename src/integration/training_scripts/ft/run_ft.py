#!/usr/bin/env python3
"""
FT Pipeline Training Script
============================

Automated pipeline for Fine-Tuning using LoRA and its variants.

This script trains a model using parameter-efficient fine-tuning:
  - Standard LoRA: Low-rank adaptation
  - QLoRA: 4-bit quantized base + LoRA adapters
  - DoRA: Magnitude+direction decomposition
  - LoRA+: Distinct learning rates for A/B matrices

Key Difference from KD Pipeline:
- FT: Single model, pure supervised learning (CE loss only)
- KD: Teacher+Student, distillation loss (α·KL + (1-α)·CE)

Usage:
    python run_ft.py --config config.yaml
    python run_ft.py --config config.yaml --method qlora
    python run_ft.py --config config.yaml --eval-only

Requirements:
    - Base model (HF model ID or local path)
    - Sufficient GPU memory (8GB+ for QLoRA, 16GB+ for LoRA)
    - datasets, transformers, peft, torch, bitsandbytes (for QLoRA)
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add repository root to path for imports
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

from hivemind_toolkit.ft import FTManager, FTConfig
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)


def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration"""
    log_level = getattr(logging, config.get("logging", {}).get("log_level", "INFO"))

    # Create log directory
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # Log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ft_training_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info(f"Logging to: {log_file}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_ft_config(config_dict: Dict[str, Any]) -> FTConfig:
    """
    Create FTConfig from YAML dictionary

    Args:
        config_dict: Dictionary from YAML config

    Returns:
        FTConfig object
    """
    training = config_dict.get("training", {})
    lora = config_dict.get("lora", {})
    data = config_dict.get("data", {})
    evaluation = config_dict.get("evaluation", {})
    checkpointing = config_dict.get("checkpointing", {})
    logging_config = config_dict.get("logging", {})
    device_config = config_dict.get("device", {})
    quantization = config_dict.get("quantization", {})

    ft_config = FTConfig(
        # Model path
        model_path=config_dict["model_path"],
        # Training
        learning_rate=training.get("learning_rate", 1e-4),
        batch_size=training.get("batch_size", 4),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 8),
        num_epochs=training.get("num_epochs", 3),
        max_steps=training.get("max_steps"),
        warmup_steps=training.get("warmup_steps", 100),
        max_grad_norm=training.get("max_grad_norm", 1.0),
        weight_decay=training.get("weight_decay", 0.01),
        # LoRA
        use_lora=lora.get("enabled", True),
        lora_r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        lora_target_modules=lora.get("target_modules"),
        # LoRA variants
        use_dora=lora.get("use_dora", False),
        use_rslora=lora.get("use_rslora", False),
        # Quantization (for QLoRA)
        use_quantization=quantization.get("enabled", False),
        quantization_type=quantization.get("type", "nf4"),
        bnb_4bit_compute_dtype=quantization.get("compute_dtype", "bfloat16"),
        bnb_4bit_use_double_quant=quantization.get("double_quant", True),
        # Data
        dataset_name=data.get("dataset_name", "c4"),
        dataset_split=data.get("dataset_split", "train"),
        max_samples=data.get("max_samples"),
        max_length=data.get("max_length", 512),
        # Evaluation
        eval_steps=evaluation.get("eval_steps", 500),
        eval_samples=evaluation.get("eval_samples", 100),
        compute_perplexity=evaluation.get("compute_perplexity", True),
        # Checkpointing
        output_dir=checkpointing.get("output_dir", "./checkpoints/ft"),
        save_steps=checkpointing.get("save_steps", 1000),
        save_total_limit=checkpointing.get("save_total_limit", 3),
        resume_from_checkpoint=checkpointing.get("resume_from_checkpoint"),
        # Logging
        logging_steps=logging_config.get("logging_steps", 10),
        log_level=logging_config.get("log_level", "INFO"),
        # Device
        device=device_config.get("device", "auto"),
        fp16=training.get("fp16", False),
        bf16=training.get("bf16", True),
        # Advanced
        seed=config_dict.get("advanced", {}).get("seed", 42),
        dataloader_num_workers=data.get("num_workers", 0),
        gradient_checkpointing=training.get("gradient_checkpointing", False),
    )

    return ft_config


def load_training_data(
    config: Dict[str, Any],
    tokenizer,
) -> DataLoader:
    """
    Load and prepare training data.

    Supports multiple dataset formats:
      - Plain text datasets with 'text' or 'content' columns (e.g. C4, WikiText)
      - Chat/instruction datasets with 'messages' column (e.g. local JSONL agent files)
      - Local JSONL files (dataset_name ends with .jsonl)

    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer for preprocessing

    Returns:
        DataLoader for training
    """
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    dataset_name = data_config.get("dataset_name", "c4")
    dataset_split = data_config.get("dataset_split", "train")
    dataset_subset = data_config.get("dataset_subset", None)
    max_samples = data_config.get("max_samples")
    max_length = data_config.get("max_length", 512)
    batch_size = training_config.get("batch_size", 4)
    num_workers = data_config.get("num_workers", 0)

    logger.info(f"Loading training dataset: {dataset_name} ({dataset_split})")

    # Load dataset
    if dataset_name.endswith(".jsonl") or dataset_name.endswith(".json"):
        dataset = load_dataset("json", data_files=dataset_name, split="train")
    elif dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split=dataset_split, streaming=True)
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=dataset_split)
    else:
        kwargs = {}
        if dataset_subset:
            kwargs["name"] = dataset_subset
        dataset = load_dataset(dataset_name, split=dataset_split, **kwargs)

    # Limit samples
    if max_samples is not None:
        if hasattr(dataset, '__len__'):
            max_samples = min(max_samples, len(dataset))
        if hasattr(dataset, "take"):
            dataset = dataset.take(max_samples)
        else:
            dataset = dataset.select(range(max_samples))

    # Detect format: chat (messages) vs plain text
    #sample_cols = dataset.column_names if hasattr(dataset, "column_names") else []
   #has_messages = "messages" in sample_cols

   # Detect dataset format
    sample_cols = dataset.column_names if hasattr(dataset, "column_names") else []

    has_messages = "messages" in sample_cols
    has_instruction_format = (
    "instruction" in sample_cols
    and "input" in sample_cols
    and "output" in sample_cols
)

    # Tokenization function
    def tokenize_function(examples):
        if has_messages:
            # Chat format: apply chat template
            texts = []
            for msgs in examples["messages"]:
                text = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(text)

        elif has_instruction_format:
            # Pattern 1 traceability format:
            # instruction + input{higher_level_requirement, lower_level_requirement} + output
            texts = []

            for instruction, input_obj, output in zip(
                examples["instruction"],
                examples["input"],
                examples["output"]
            ):
                higher_req = input_obj.get("higher_level_requirement", "")
                lower_req = input_obj.get("lower_level_requirement", "")

                prompt = (
                    f"{instruction}\n\n"
                    f"Higher-level requirement:\n{higher_req}\n\n"
                    f"Lower-level requirement:\n{lower_req}\n\n"
                    f"Answer:"
                )

                text = f"{prompt} {output}"
                texts.append(text)

        elif "text" in examples:
            texts = examples["text"]

        elif "content" in examples:
            texts = examples["content"]

        else:
            raise ValueError(
                "Dataset must have 'messages', 'text', 'content', "
                "or 'instruction'/'input'/'output' fields"
            )

        # Filter out empty texts
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return {"input_ids": [], "attention_mask": []}

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        return tokenized

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    cols_to_remove = dataset.column_names if hasattr(dataset, "column_names") else []
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=cols_to_remove,
    )

    # Data collator (handles padding and labels)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM (not masked LM)
    )

    # Create DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
    )

    logger.info("Training data loaded")
    return dataloader


def load_eval_data(
    config: Dict[str, Any],
    tokenizer,
) -> list:
    """
    Load evaluation data.

    Supports:
    - local JSON/JSONL Pattern 1 traceability files
    - chat messages format
    - text/content format
    - default HuggingFace datasets such as c4 and wikitext

    Returns:
        List of evaluation texts
    """
    eval_config = config.get("evaluation", {})
    eval_samples = eval_config.get("eval_samples", 100)
    eval_dataset = eval_config.get("eval_dataset", "wikitext")
    eval_split = eval_config.get("eval_split", "validation")

    logger.info(f"Loading evaluation dataset: {eval_dataset} ({eval_split})")

    eval_texts = []

    # -----------------------------
    # Local JSON / JSONL evaluation file
    # -----------------------------
    if eval_dataset.endswith(".jsonl") or eval_dataset.endswith(".json"):
        dataset = load_dataset("json", data_files=eval_dataset, split="train")

        if eval_samples is not None:
            dataset = dataset.select(range(min(eval_samples, len(dataset))))

        sample_cols = dataset.column_names

        has_messages = "messages" in sample_cols
        has_instruction_format = (
            "instruction" in sample_cols
            and "input" in sample_cols
            and "output" in sample_cols
        )

        for example in dataset:
            if has_messages:
                text = tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False
                )

            elif has_instruction_format:
                instruction = example.get("instruction", "")
                input_obj = example.get("input", {})
                output = example.get("output", "")

                higher_req = input_obj.get("higher_level_requirement", "")
                lower_req = input_obj.get("lower_level_requirement", "")

                prompt = (
                    f"{instruction}\n\n"
                    f"Higher-level requirement:\n{higher_req}\n\n"
                    f"Lower-level requirement:\n{lower_req}\n\n"
                    f"Answer:"
                )

                text = f"{prompt} {output}"

            elif "text" in example:
                text = example["text"]

            elif "content" in example:
                text = example["content"]

            else:
                raise ValueError(
                    "Evaluation dataset must have 'messages', 'text', 'content', "
                    "or 'instruction'/'input'/'output' fields"
                )

            if text and text.strip():
                eval_texts.append(text)

        logger.info(f"Loaded {len(eval_texts)} local evaluation samples")
        return eval_texts

    # -----------------------------
    # HuggingFace evaluation datasets
    # -----------------------------
    if eval_dataset == "c4":
        dataset = load_dataset("allenai/c4", "en", split=eval_split, streaming=True)
        dataset = dataset.take(eval_samples)

    elif eval_dataset == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=eval_split)
        dataset = dataset.select(range(min(eval_samples, len(dataset))))

    else:
        dataset = load_dataset(eval_dataset, split=eval_split)
        dataset = dataset.select(range(min(eval_samples, len(dataset))))

    for example in dataset:
        if "text" in example:
            text = example["text"]
        elif "content" in example:
            text = example["content"]
        else:
            continue

        if text and text.strip():
            eval_texts.append(text)

    logger.info(f"Loaded {len(eval_texts)} evaluation samples")
    return eval_texts

def evaluate_baseline(manager, eval_texts: list, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate base model (without LoRA adapters) for baseline comparison

    This must be called AFTER load_model() but BEFORE initialize_trainer()
    to evaluate the unmodified base model.

    Args:
        manager: FTManager with loaded base model
        eval_texts: Evaluation texts
        config: Configuration dictionary

    Returns:
        Dictionary with baseline metrics
    """
    logger.info("\n" + "="*70)
    logger.info("BASELINE EVALUATION (Pre-FT)")
    logger.info("="*70)
    logger.info("Evaluating base model before fine-tuning...")

    # Use manager's evaluate method (works on current model state)
    baseline_results = manager.evaluate(eval_texts)

    logger.info(f"Baseline Perplexity: {baseline_results['perplexity']:.2f}")
    logger.info(f"Baseline Loss: {baseline_results['loss']:.4f}")
    logger.info("="*70 + "\n")

    return baseline_results


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="FT Pipeline - Fine-Tuning with LoRA variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="FT method to use (overrides config): lora, qlora, dora",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (no training)",
    )

    parser.add_argument(
        "--comprehensive-eval",
        action="store_true",
        help="Run comprehensive evaluation (perplexity + speed + memory profiling)",
    )

    return parser.parse_args()


def main():
    """Main FT pipeline execution"""
    args = parse_args()

    print("\n" + "=" * 70)
    print("FT PIPELINE - Parameter-Efficient Fine-Tuning")
    print("=" * 70 + "\n")

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)

    # Setup logging
    setup_logging(config)
    logger.info("FT Pipeline started")
    logger.info(f"Configuration: {args.config}")

    # Set seed
    seed = config.get("advanced", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Create FT config
    ft_config = create_ft_config(config)

    # Override method if specified
    ft_method = args.method if args.method else config.get("method", "lora")
    logger.info(f"FT Method: {ft_method}")

    # Initialize FT Manager
    logger.info("\nInitializing FT Manager...")
    manager = FTManager(ft_config)

    # Load model (single model, not teacher+student)
    manager.load_model()

    # Load evaluation data (needed for baseline evaluation)
    eval_texts = load_eval_data(config, manager.tokenizer)

    # Eval-only mode
    if args.eval_only:
        logger.info("\n[EVAL-ONLY MODE]")
        results = manager.evaluate(eval_texts)

        logger.info("\nEvaluation Results:")
        logger.info(f"  Perplexity: {results['perplexity']:.2f}")
        logger.info(f"  Loss: {results['loss']:.4f}")
        return

    # Baseline evaluation (BEFORE applying LoRA adapters)
    baseline_results = evaluate_baseline(manager, eval_texts, config)

    # Initialize trainer (applies LoRA adapters)
    manager.initialize_trainer(method=ft_method)

    # Load training data
    train_dataloader = load_training_data(config, manager.tokenizer)

    # Train
    logger.info("\nStarting training...")
    history = manager.train(
        train_dataloader=train_dataloader,
        eval_data=eval_texts,
    )

    # Final evaluation
    logger.info("\nRunning final evaluation...")

    if args.comprehensive_eval:
        logger.info("Running comprehensive evaluation (perplexity + speed + memory)...")
        final_results = manager.evaluate_comprehensive(
            eval_texts,
            benchmark_speed=True,
            profile_memory=True,
            save_results=True,
            output_path=os.path.join(ft_config.output_dir, "final_evaluation.json")
        )

        logger.info("\nFinal Results (Comprehensive):")
        logger.info(f"  Perplexity: {final_results['perplexity']['perplexity']:.2f}")
        logger.info(f"  Loss: {final_results['perplexity']['loss']:.4f}")

        if "speed" in final_results:
            logger.info(f"\nSpeed Benchmarks:")
            logger.info(f"  Avg time: {final_results['speed']['avg_time_ms']:.2f} ms")
            logger.info(f"  Throughput: {final_results['speed']['throughput_per_sec']:.2f} iterations/sec")
            if "tokens_per_sec" in final_results["speed"]:
                logger.info(f"  Tokens/sec: {final_results['speed']['tokens_per_sec']:.2f}")

        if "memory" in final_results:
            logger.info(f"\nMemory Profile:")
            logger.info(f"  Model size: {final_results['memory']['param_size_gb']:.2f} GB")
            logger.info(f"  Total params: {final_results['memory']['total_params_millions']:.1f}M")
            logger.info(f"  Trainable params: {final_results['memory']['trainable_params_millions']:.1f}M")
    else:
        # Standard evaluation (current behavior)
        final_results = manager.evaluate(eval_texts)

        logger.info("\nFinal Results:")
        logger.info(f"  Perplexity: {final_results['perplexity']:.2f}")
        logger.info(f"  Loss: {final_results['loss']:.4f}")

    # Comparison with baseline
    logger.info("\n" + "="*70)
    logger.info("BASELINE vs FINE-TUNED COMPARISON")
    logger.info("="*70)

    # Extract perplexity values (handle both standard and comprehensive formats)
    if args.comprehensive_eval:
        final_ppl = final_results['perplexity']['perplexity']
        final_loss = final_results['perplexity']['loss']
    else:
        final_ppl = final_results['perplexity']
        final_loss = final_results['loss']

    # Perplexity comparison
    ppl_improvement = (
        (baseline_results['perplexity'] - final_ppl) /
        baseline_results['perplexity'] * 100
    )
    logger.info(f"Baseline Perplexity:    {baseline_results['perplexity']:>8.2f}")
    logger.info(f"Fine-Tuned Perplexity:  {final_ppl:>8.2f}")
    logger.info(f"Improvement:            {ppl_improvement:>7.1f}%")

    # Loss comparison
    loss_improvement = (
        (baseline_results['loss'] - final_loss) /
        baseline_results['loss'] * 100
    )
    logger.info(f"\nBaseline Loss:          {baseline_results['loss']:>8.4f}")
    logger.info(f"Fine-Tuned Loss:        {final_loss:>8.4f}")
    logger.info(f"Improvement:            {loss_improvement:>7.1f}%")

    # Interpretation
    if ppl_improvement > 0:
        logger.info(f"\nFine-tuning IMPROVED model by {ppl_improvement:.1f}%")
    elif ppl_improvement > -5:
        logger.info(f"\n[~] Fine-tuning had minimal effect ({ppl_improvement:.1f}%)")
    else:
        logger.info(f"\n[!] Fine-tuning DEGRADED model by {abs(ppl_improvement):.1f}%")

    logger.info("="*70)

    # Save final model
    final_output = os.path.join(ft_config.output_dir, "final_model")
    logger.info(f"\nSaving final model to: {final_output}")
    manager.trainer.save_checkpoint(final_output)

    logger.info("\nFT Pipeline complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        sys.exit(1)
