#!/usr/bin/env python3
"""
Base classes for Fine-Tuning (FT) Pipeline
===========================================

Abstract base classes and configuration for parameter-efficient fine-tuning
using LoRA and its variants.

Key Difference from KD Pipeline:
- FT: Single model, pure supervised learning (CE loss only)
- KD: Teacher+Student, distillation loss (α·KL + (1-α)·CE)

Academic References:
  - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
    arXiv:2106.09685 - Original LoRA paper
  - Dettmers et al., "QLoRA: Efficient Fine-Tuning of Quantized LLMs" (2023)
    arXiv:2305.14314 - 4-bit quantized training
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from prefect import task
from prefect.cache_policies import NO_CACHE
import mlflow


@dataclass
class FTConfig:
    """Configuration for Fine-Tuning pipeline"""

    # Model configuration
    model_path: str

    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    max_steps: Optional[int] = None
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # LoRA variant-specific
    use_dora: bool = False  # DoRA: Magnitude+direction decomposition
    use_rslora: bool = False  # Rank-stabilized LoRA

    # QLoRA configuration (4-bit quantization)
    use_quantization: bool = False
    quantization_type: str = "nf4"  # nf4 or fp4
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    # Data configuration
    dataset_name: str = "c4"
    dataset_split: str = "train"
    max_samples: Optional[int] = None
    max_length: int = 512

    # Evaluation
    eval_steps: int = 500
    eval_samples: int = 100
    compute_perplexity: bool = True

    # Checkpointing
    output_dir: str = "./checkpoints/ft"
    save_steps: int = 1000
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None

    # Logging
    logging_steps: int = 10
    log_level: str = "INFO"

    # Device and precision
    device: str = "auto"
    fp16: bool = False
    bf16: bool = True

    # Advanced options
    seed: int = 42
    dataloader_num_workers: int = 0
    gradient_checkpointing: bool = False

    def __post_init__(self):
        """Validate and set defaults"""
        # Set default target modules if not specified
        if self.lora_target_modules is None:
            # Default: Full attention (Q, K, V, O projections)
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj"
            ]

        # Resolve "auto" device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use bf16 if available for better stability
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.bf16 = True
            self.fp16 = False

        # Validate parameters
        assert 1e-6 <= self.learning_rate <= 1e-2, f"Invalid learning_rate: {self.learning_rate}"
        assert 1 <= self.lora_r <= 256, f"Invalid lora_r: {self.lora_r}"
        assert 0.0 <= self.lora_dropout <= 1.0, f"Invalid lora_dropout: {self.lora_dropout}"
        assert self.batch_size > 0, f"Invalid batch_size: {self.batch_size}"
        assert self.gradient_accumulation_steps > 0, f"Invalid gradient_accumulation_steps"

        # Warn about incompatible options
        if self.use_quantization and self.use_dora:
            print("[WARNING] DoRA with quantization may be unstable. Use with caution.")


class BaseFTTrainer(ABC):
    """
    Abstract base class for Fine-Tuning trainers.

    All FT methods should inherit from this class and implement:
      - prepare_model() - Apply LoRA/adapters to model
      - compute_loss() - Compute training loss (pure CE, no KD)
      - get_method_name() - Return method identifier

    Key Difference from KD:
    - No teacher model (single model only)
    - No temperature/alpha parameters (no distillation)
    - Pure supervised learning (CE loss only)
    """

    def __init__(
        self,
        config: FTConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Initialize FT trainer

        Args:
            config: FTConfig with training parameters
            model: Base model to fine-tune
            tokenizer: Tokenizer for the model
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Move model to device (if not using device_map)
        if not hasattr(self.model, "hf_device_map"):
            self.model = self.model.to(config.device)

        print(f"BaseFTTrainer initialized")
        print(f"  Method: {self.get_method_name()}")
        print(f"  Model: {config.model_path}")
        print(f"  Device: {config.device}")
        print(f"  LoRA rank: {config.lora_r}")
        print(f"  LoRA alpha: {config.lora_alpha}")


    @task(cache_key_fn=lambda *_: "prepare_model", cache_policy=NO_CACHE)
    @abstractmethod
    def prepare_model(self) -> PreTrainedModel:
        """
        Prepare model for training by applying LoRA/adapters

        Returns:
            Model with adapters applied
        """
        pass


    @task(cache_key_fn=lambda *_: "compute_loss", cache_policy=NO_CACHE)
    @abstractmethod
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss

        Args:
            batch: Dictionary with input_ids, attention_mask, labels

        Returns:
            Tuple of (loss, metrics_dict)

        Note: Pure supervised learning (CE loss only, no KD loss)
        """
        pass


    @task(cache_key_fn=lambda *_: "get_method_name", cache_policy=NO_CACHE)
    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of the FT method (e.g., 'lora', 'qlora', 'dora')"""
        pass


    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform a single training step

        Args:
            batch: Dictionary with input_ids, attention_mask, labels

        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.config.device) for k, v in batch.items()}

        # Compute loss
        loss, metrics = self.compute_loss(batch)

        # Add step info
        metrics["step"] = self.global_step
        metrics["epoch"] = self.epoch

        return loss, metrics


    def save_checkpoint(self, output_dir: str):
        """Save model checkpoint to local directory and log to MLflow"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)

        # Save model and tokenizer locally
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "method": self.get_method_name(),
        }
        state_path = os.path.join(output_dir, "trainer_state.pt")
        torch.save(state, state_path)
        print(f"Checkpoint saved to: {output_dir}")
        
        # Log checkpoint to MLflow if active run exists
        try:
            active_run = mlflow.active_run()
            if active_run is not None:
                # Log the entire checkpoint directory as an artifact
                mlflow.log_artifacts(output_dir, artifact_path=f"checkpoints/step_{self.global_step}")
                
                # Log checkpoint metrics
                mlflow.log_metrics({
                    "checkpoint_step": self.global_step,
                    "checkpoint_epoch": self.epoch,
                    "checkpoint_best_loss": self.best_loss,
                }, step=self.global_step)
                
                print(f"Checkpoint logged to MLflow run: {active_run.info.run_id}")
            else:
                print("No active MLflow run. Checkpoint saved locally only.")
        except Exception as e:
            print(f"Warning: Failed to log checkpoint to MLflow: {e}")
            print("Checkpoint saved locally successfully.")


    @task(cache_key_fn=lambda *_: "load_checkpoint", cache_policy=NO_CACHE)
    def load_checkpoint(self, checkpoint_dir: str = None, run_id: str = None, artifact_path: str = None):
        """
        Load checkpoint from MLflow or local directory
        
        Args:
            checkpoint_dir: Local directory path to load from (optional if using MLflow)
            run_id: MLflow run ID to load checkpoint from (optional)
            artifact_path: Path to artifact within MLflow run (e.g., 'checkpoints/step_1000')
        """
        import os
        import tempfile
        
        loaded_from_mlflow = False
        temp_dir = None
        
        # Try to load from MLflow if run_id is provided
        if run_id is not None:
            try:
                print(f"Attempting to load checkpoint from MLflow run: {run_id}")
                
                # Default artifact path if not specified
                if artifact_path is None:
                    artifact_path = "checkpoints"
                
                # Download artifacts from MLflow to a temporary directory
                temp_dir = tempfile.mkdtemp(prefix="mlflow_checkpoint_")
                mlflow_client = mlflow.tracking.MlflowClient()
                mlflow_client.download_artifacts(run_id, artifact_path, temp_dir)
                
                # Set checkpoint_dir to the downloaded location
                checkpoint_dir = os.path.join(temp_dir, artifact_path)
                loaded_from_mlflow = True
                print(f"Checkpoint downloaded from MLflow to: {checkpoint_dir}")
                
            except Exception as e:
                print(f"Warning: Failed to load checkpoint from MLflow: {e}")
                if checkpoint_dir is None:
                    raise ValueError("No valid checkpoint source: MLflow download failed and no local checkpoint_dir provided")
                print(f"Falling back to local checkpoint: {checkpoint_dir}")
        
        # Load from local directory (either provided or downloaded from MLflow)
        if checkpoint_dir is None:
            raise ValueError("Either checkpoint_dir or run_id must be provided")
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Load training state
        state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.config.device)
            self.global_step = state.get("global_step", 0)
            self.epoch = state.get("epoch", 0)
            self.best_loss = state.get("best_loss", float('inf'))
            
            source = "MLflow" if loaded_from_mlflow else "local directory"
            print(f"Checkpoint loaded from {source}")
            print(f"Resumed from step {self.global_step}, epoch {self.epoch}")
        else:
            print(f"Warning: trainer_state.pt not found in {checkpoint_dir}")
        
        # Clean up temporary directory if it was created
        if temp_dir is not None and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Failed to clean up temporary directory: {e}")


    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state"""
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "method": self.get_method_name(),
        }
