#!/usr/bin/env python3
"""
FT Manager - Orchestrates Fine-Tuning Pipeline
===============================================

Manages model loading, trainer initialization, and training execution for the FT pipeline.

Key Responsibilities:
- Load single model (NOT teacher+student like KD)
- Initialize trainer for specific LoRA variant
- Execute training loop
- Handle evaluation and checkpointing
"""

import os
import torch
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging

from .base import FTConfig, BaseFTTrainer
from .methods import LoRATrainer, QLoRATrainer, DoRATrainer
from hivemind_toolkit.evaluation import Evaluator


from prefect import task
from prefect.cache_policies import NO_CACHE
import mlflow


logger = logging.getLogger(__name__)


class FTManager:
    """
    Manager for Fine-Tuning pipeline orchestration

    Handles:
    - Model loading (single model, not teacher+student)
    - Trainer initialization for specific LoRA variant
    - Training loop execution
    - Evaluation and checkpointing
    """

    def __init__(self, config: FTConfig):
        """
        Initialize FT Manager

        Args:
            config: FTConfig with training parameters
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

        print("FTManager initialized")
        print(f"Model: {config.model_path}")
        print(f"Method: Will be set during initialize_trainer()")


    @task(cache_policy=NO_CACHE)
    def load_model(self):
        """
        Load model and tokenizer

        Note: Loads single model only (no teacher like KD pipeline)
        """
        print(f"Loading model: {self.config.model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with optional quantization
        if self.config.use_quantization:
            print("Loading model with 4-bit quantization (QLoRA mode)")
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.quantization_type,
                bnb_4bit_compute_dtype=getattr(
                    torch,
                    self.config.bnb_4bit_compute_dtype
                ),
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                quantization_config=bnb_config,
                device_map={"":0},
                trust_remote_code=True,
            )
        else:
            # Load in full precision (FP16/BF16)
            dtype = torch.bfloat16 if self.config.bf16 else (
                torch.float16 if self.config.fp16 else torch.float32
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=dtype,
                device_map={"":0} if self.config.device == "cuda" else None,
                trust_remote_code=True,
            )

        print("Model loaded successfully")
        print(f"  Parameters: {self.model.num_parameters():,}")
        print(f"  Dtype: {self.model.dtype}")


    @task(cache_policy=NO_CACHE)
    def initialize_trainer(
        self,
        method: str = "lora",
        **method_params
    ):
        """
        Initialize trainer for specific LoRA variant

        Args:
            method: FT method to use ('lora', 'qlora', 'dora', 'loraplus')
            **method_params: Additional method-specific parameters
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Must call load_model() before initialize_trainer()")

        print(f"Initializing trainer: {method}")

        # Method registry
        method_map = {
            "lora": LoRATrainer,
            "qlora": QLoRATrainer,
            "dora": DoRATrainer,
            # Future: "loraplus": LoRAPlusTrainer,
        }

        if method not in method_map:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Available: {list(method_map.keys())}"
            )

        # Instantiate trainer
        trainer_class = method_map[method]
        self.trainer = trainer_class(
            config=self.config,
            model=self.model,
            tokenizer=self.tokenizer,
            **method_params
        )

        print(f"Trainer initialized: {self.trainer.get_method_name()}")


    @task(cache_policy=NO_CACHE)   
    def train(
        self,
        train_dataloader,
        eval_data: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Execute training loop

        Args:
            train_dataloader: DataLoader for training data
            eval_data: Optional evaluation data (list of texts)

        Returns:
            Training history dictionary
        """
        if self.trainer is None:
            raise RuntimeError("Must call initialize_trainer() before train()")

        print("\nStarting training...")
        print("=" * 70)

        # Prepare model (apply LoRA adapters)
        self.model = self.trainer.prepare_model()

        # Re-enable input require grads after PeftModel wrapping —
        # get_peft_model can reset the hook set by prepare_model_for_kbit_training
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()
        
        # Enable memory-efficient attention if available
        if hasattr(self.model.config, 'use_flash_attention_2'):
            self.model.config.use_flash_attention_2 = False  # Disable if causing issues

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Setup learning rate scheduler
        from transformers import get_linear_schedule_with_warmup

        # Compute total training steps (handle streaming datasets like C4)
        if self.config.max_steps is not None:
            # User explicitly set max_steps - use it
            total_steps = self.config.max_steps
            print(f"Using max_steps from config: {total_steps}")
        elif hasattr(train_dataloader.dataset, '__len__'):
            # Regular dataset with known length
            total_steps = (
                len(train_dataloader) * self.config.num_epochs //
                self.config.gradient_accumulation_steps
            )
            print(f"Computed total_steps from dataset length: {total_steps}")
        elif self.config.max_samples is not None:
            # Estimate from max_samples for streaming datasets (like C4)
            steps_per_epoch = (
                self.config.max_samples //
                (self.config.batch_size * self.config.gradient_accumulation_steps)
            )
            total_steps = steps_per_epoch * self.config.num_epochs
            print(
                f"Estimated total_steps for streaming dataset: {total_steps} "
                f"({steps_per_epoch} steps/epoch × {self.config.num_epochs} epochs)"
            )
        else:
            # Fallback: Use a reasonable default
            total_steps = 1000
            print(
                f"Cannot determine dataset length for streaming dataset. "
                f"Using default total_steps={total_steps}. "
                f"Set max_steps or max_samples in config for accurate scheduler."
            )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        # Training history
        history = {
            "train_loss": [],
            "eval_perplexity": [],
            "learning_rate": [],
        }

        # Training loop
        global_step = 0
        best_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            epoch_loss = 0.0
            optimizer.zero_grad()

            # Calculate expected optimizer steps for this epoch
            # (handle both regular and streaming datasets)
            if hasattr(train_dataloader.dataset, '__len__'):
                batches_this_epoch = len(train_dataloader)
            elif self.config.max_samples is not None:
                batches_this_epoch = self.config.max_samples // self.config.batch_size
            else:
                batches_this_epoch = None

            if batches_this_epoch is not None:
                expected_steps_this_epoch = (
                    batches_this_epoch // self.config.gradient_accumulation_steps
                )
            else:
                expected_steps_this_epoch = None

            # Create progress bar counting optimizer steps (not batches)
            progress_bar = tqdm(
                total=expected_steps_this_epoch,
                desc=f"Training Epoch {epoch + 1}",
                unit="step"
            )

            for step, batch in enumerate(train_dataloader):
                # Check if we've reached max_steps
                if self.config.max_steps is not None and global_step >= self.config.max_steps:
                    print(f"\nReached max_steps ({self.config.max_steps}). Stopping training.")
                    break
                
                # Check if we've processed max_samples (for streaming datasets)
                if self.config.max_samples is not None:
                    samples_processed = step * self.config.batch_size
                    if samples_processed >= self.config.max_samples:
                        print(f"\nReached max_samples ({self.config.max_samples}). Stopping epoch.")
                        break
                
                # Training step
                loss, metrics = self.trainer.train_step(batch)

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps

                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Clear GPU cache periodically to prevent fragmentation
                    if global_step % 10 == 0:
                        torch.cuda.empty_cache()

                    # Update progress bar on EVERY optimizer step
                    avg_loss = epoch_loss / (step + 1)
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "global": f"{global_step}/{total_steps}"
                    })
                    progress_bar.update(1)  # Increment by 1 optimizer step

                    # Log to history (only on logging_steps)
                    if global_step % self.config.logging_steps == 0:
                        history["train_loss"].append(avg_loss)
                        history["learning_rate"].append(scheduler.get_last_lr()[0])

                    # Evaluation
                    if eval_data and global_step % self.config.eval_steps == 0:
                        eval_results = self.evaluate(eval_data)
                        history["eval_perplexity"].append(
                            eval_results["perplexity"]
                        )
                        print(
                            f"\nStep {global_step} - "
                            f"Eval Perplexity: {eval_results['perplexity']:.2f}"
                        )

                    # Checkpointing
                    if global_step % self.config.save_steps == 0:
                        checkpoint_dir = os.path.join(
                            self.config.output_dir,
                            f"checkpoint-{global_step}"
                        )
                        self.trainer.save_checkpoint(checkpoint_dir)

                        # Keep only last N checkpoints
                        self._cleanup_checkpoints()

                    # Update training state
                    self.trainer.global_step = global_step
                    self.trainer.epoch = epoch

            # Close progress bar
            progress_bar.close()

            # End of epoch (use step+1 for streaming datasets)
            avg_epoch_loss = epoch_loss / (step + 1)
            print(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")

            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_dir = os.path.join(self.config.output_dir, "best_model")
                self.trainer.save_checkpoint(best_model_dir)
                print(f"Best model saved (loss: {best_loss:.4f})")
            
            # Break out of epoch loop if max_steps reached
            if self.config.max_steps is not None and global_step >= self.config.max_steps:
                print(f"Reached max_steps ({self.config.max_steps}). Ending training.")
                break

        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)

        return history


    @task(cache_policy=NO_CACHE)
    def evaluate(self, eval_texts: list) -> Dict[str, float]:
        """
        Evaluate model perplexity using unified Evaluator

        Args:
            eval_texts: List of evaluation texts

        Returns:
            Dictionary with evaluation metrics (backward compatible format)
        """
        # Create evaluator
        evaluator = Evaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.config.device
        )

        # Compute perplexity
        results = evaluator.compute_perplexity(
            texts=eval_texts,
            n_samples=self.config.eval_samples,
            max_length=self.config.max_length
        )

        # Return in same format for backward compatibility
        return {
            "loss": results["loss"],
            "perplexity": results["perplexity"],
        }


    @task(cache_policy=NO_CACHE)
    def evaluate_comprehensive(
        self,
        eval_texts: list,
        benchmark_speed: bool = False,
        profile_memory: bool = False,
        save_results: bool = False,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with all metrics

        Useful for final evaluation or detailed analysis. Includes perplexity,
        and optionally speed benchmarking and memory profiling.

        Args:
            eval_texts: List of evaluation texts
            benchmark_speed: Whether to benchmark inference speed
            profile_memory: Whether to profile memory usage
            save_results: Whether to save results to JSON file
            output_path: Path to save results (default: output_dir/evaluation_results.json)

        Returns:
            Dictionary with comprehensive evaluation results including:
                - perplexity: Perplexity metrics
                - speed: Speed benchmarks (if requested)
                - memory: Memory profile (if requested)

        Example:
            >>> results = manager.evaluate_comprehensive(
            ...     eval_texts,
            ...     benchmark_speed=True,
            ...     profile_memory=True,
            ...     save_results=True
            ... )
        """
        # Create evaluator
        evaluator = Evaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.config.device
        )

        # Set default output path
        if save_results and output_path is None:
            output_path = os.path.join(self.config.output_dir, "evaluation_results.json")

        # Run comprehensive evaluation
        results = evaluator.evaluate_comprehensive(
            texts=eval_texts,  # Pass texts directly
            n_samples=self.config.eval_samples,
            benchmark_speed=benchmark_speed,
            profile_memory=profile_memory,
            save_results=save_results,
            output_path=output_path,
        )

        return results


    def _cleanup_checkpoints(self):
        """Keep only last N checkpoints"""
        if self.config.save_total_limit is None:
            return

        # Get all checkpoint directories
        checkpoints = []
        for name in os.listdir(self.config.output_dir):
            if name.startswith("checkpoint-"):
                checkpoints.append(name)

        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))

        # Remove old checkpoints
        while len(checkpoints) > self.config.save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            checkpoint_path = os.path.join(self.config.output_dir, old_checkpoint)
            import shutil
            shutil.rmtree(checkpoint_path, ignore_errors=True)
            print(f"Removed old checkpoint: {old_checkpoint}")
