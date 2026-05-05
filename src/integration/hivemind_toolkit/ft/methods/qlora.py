#!/usr/bin/env python3
"""
QLoRA Trainer
=============

Implements QLoRA (Quantized LoRA) for memory-efficient fine-tuning.

Method:
  - Base model: 4-bit NF4 quantization
  - LoRA adapters: Full precision (FP16/BF16)
  - Memory savings: ~4× vs full fine-tuning

Technical Details:
  1. NF4 Quantization: Information-theoretically optimal 4-bit format
  2. Double Quantization: Quantize the quantization constants
  3. Paged Optimizers: Handle memory spikes via CPU offload

Academic References:
  - Dettmers et al., "QLoRA: Efficient Fine-Tuning of Quantized LLMs" (2023)
    arXiv:2305.14314 - 4-bit quantized training
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging

from ..base import BaseFTTrainer, FTConfig

logger = logging.getLogger(__name__)


class QLoRATrainer(BaseFTTrainer):
    """
    QLoRA trainer for memory-efficient fine-tuning

    Key Features:
    - 4-bit quantized base model (NF4 format)
    - LoRA adapters in full precision
    - ~4× memory savings vs full fine-tuning
    - Enables training 65B models on single 48GB GPU
    """

    def __init__(
        self,
        config: FTConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Initialize QLoRA trainer

        Args:
            config: FTConfig with training parameters
            model: Base model (should be loaded with 4-bit quantization)
            tokenizer: Tokenizer

        Note:
        Model should be loaded with BitsAndBytesConfig before passing here.
        See FTManager.load_model() for proper loading.
        """
        super().__init__(config, model, tokenizer)

        # Verify model is quantized
        if not hasattr(model, "quantization_config"):
            logger.warning(
                "[WARNING] Model does not have quantization_config. "
                "QLoRA expects a 4-bit quantized model. "
                "Set use_quantization=True in FTConfig."
            )

        logger.info("Initialized QLoRA Trainer")
        logger.info(f"  Base model: 4-bit quantized")
        logger.info(f"  LoRA adapters: Full precision")
        logger.info(f"  Rank (r): {config.lora_r}")
        logger.info(f"  Alpha: {config.lora_alpha}")

    def prepare_model(self) -> PreTrainedModel:
        """
        Apply LoRA adapters to quantized model

        Returns:
            Model with LoRA adapters applied
        """
        logger.info("Applying LoRA adapters to quantized model...")

        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            raise ImportError(
                "The 'peft' library is required for QLoRA.\n"
                "Install it with: pip install peft>=0.7.0"
            )

        # Prepare model for k-bit training
        # This enables gradient computation through quantized layers
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.gradient_checkpointing
        )

        # Create LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        logger.info("LoRA adapters applied to quantized model")
        logger.info("  Base model: 4-bit (frozen)")
        logger.info("  LoRA adapters: Full precision (trainable)")

        return self.model

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss (pure supervised learning)

        Args:
            batch: Dictionary with input_ids, attention_mask, labels

        Returns:
            Tuple of (loss, metrics_dict)

        Note:
        Same as standard LoRA - pure cross-entropy loss.
        The only difference is the base model is quantized.
        """
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # Get loss (cross-entropy with labels)
        loss = outputs.loss

        # Metrics
        metrics = {
            "loss": loss.item(),
        }

        return loss, metrics

    def get_method_name(self) -> str:
        """Return method name"""
        return "qlora"
