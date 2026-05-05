#!/usr/bin/env python3
"""
Device Utilities for Multi-GPU Handling
========================================

Helper functions for properly handling device placement when models use
device_map="auto" (distributed across multiple GPUs) vs single GPU.

Key Challenge:
When using device_map="auto", HuggingFace distributes model layers across
available GPUs. Simply using next(model.parameters()).device returns the
device of the FIRST parameter, which may not be representative of where
inputs should be placed.

Solution:
Check if model uses device_map, and if so, use the embedding layer's device
as the canonical device for inputs (since embeddings are typically the first
layer that processes input tensors).
"""

import torch
from typing import Union
from prefect import task
from transformers import PreTrainedModel



@task(name="Get Model Device", retries=1, retry_delay_seconds=5)
def get_model_device(model: PreTrainedModel) -> torch.device:
    """
    Get the appropriate device for input tensors to a model.

    Handles both:
    - Single GPU models: Returns model's device
    - Multi-GPU models (device_map="auto"): Returns embedding layer's device

    Args:
        model: PreTrainedModel (may be distributed across GPUs)

    Returns:
        torch.device where input tensors should be placed

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
        >>> device = get_model_device(model)
        >>> inputs = tokenizer(text, return_tensors="pt")
        >>> inputs = {k: v.to(device) for k, v in inputs.items()}
        >>> outputs = model(**inputs)  # Works with multi-GPU!
    """
    # Check if model uses device_map (distributed across GPUs)
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Multi-GPU setup: Use embedding layer's device
        # This is where input_ids will be processed first
        try:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer is not None:
                return embedding_layer.weight.device
        except (AttributeError, RuntimeError):
            pass

    # Fallback: Single GPU or no embeddings found
    # Try to get device from model's named parameters (more efficient than iterating all)
    try:
        # Get first parameter without fully iterating
        param_name, param = next(iter(model.named_parameters()))
        return param.device
    except StopIteration:
        # No parameters (empty model?)
        return torch.device("cpu")


@task(name="Check if Model is Distributed", retries=1, retry_delay_seconds=5)
def is_model_distributed(model: PreTrainedModel) -> bool:
    """
    Check if model is distributed across multiple GPUs.

    Args:
        model: PreTrainedModel to check

    Returns:
        True if model uses device_map (distributed), False otherwise

    Example:
        >>> if is_model_distributed(model):
        ...     print("Model uses multiple GPUs")
        ... else:
        ...     print("Model on single device")
    """
    return hasattr(model, 'hf_device_map') and model.hf_device_map is not None


@task(name="Move Inputs to Device", retries=1, retry_delay_seconds=5)
def move_inputs_to_device(
    inputs: Union[torch.Tensor, dict],
    model: PreTrainedModel
) -> Union[torch.Tensor, dict]:
    """
    Move inputs to the appropriate device for the model.

    Automatically detects if model is distributed and places inputs on
    the correct device (embedding layer's device).

    Args:
        inputs: Tensor or dict of tensors to move
        model: Target model

    Returns:
        Inputs moved to appropriate device

    Example:
        >>> inputs = tokenizer(text, return_tensors="pt")
        >>> inputs = move_inputs_to_device(inputs, model)
        >>> outputs = model(**inputs)  # Always works!
    """
    device = get_model_device(model)

    if isinstance(inputs, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    else:
        return inputs


@task(name="Get Model Devices", retries=1, retry_delay_seconds=5)
def get_model_devices(model: PreTrainedModel) -> list[torch.device]:
    """
    Get list of all devices used by model parameters.

    Useful for debugging multi-GPU setups.

    Args:
        model: PreTrainedModel to inspect

    Returns:
        List of unique devices (e.g., [device('cuda:0'), device('cuda:3')])

    Example:
        >>> devices = get_model_devices(model)
        >>> print(f"Model uses devices: {devices}")
        Model uses devices: [device('cuda:0'), device('cuda:1'), device('cuda:2'), device('cuda:3')]
    """
    devices = set()
    for param in model.parameters():
        devices.add(param.device)
    return sorted(list(devices), key=lambda d: str(d))


# Convenience aliases
get_device = get_model_device
to_device = move_inputs_to_device
