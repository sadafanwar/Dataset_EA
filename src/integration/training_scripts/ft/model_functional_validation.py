"""
Model Functional Validation
============================

Loads a base model from HuggingFace and runs behavioral checks to verify
basic inference capabilities. Used as a Prefect task after model registration.
"""

import gc
import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


TEST_PROMPTS = [
    "Explain what a function is in programming.",
    "Write a simple Python hello world program.",
    "What is machine learning?",
]


def validate_model(
    model_name: str = "Qwen/Qwen3-0.6B",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> Dict[str, Any]:
    """
    Run functional validation on a model.

    If model and tokenizer are provided, uses them directly (e.g. the
    in-memory PeftModel after fine-tuning). Otherwise loads from
    model_name.

    # TODO: Remove HuggingFace loading path once all callers pass
    # in-memory model+tokenizer. Kept for standalone testing only.

    Returns: {
        "passed": bool,
        "checks_passed": int,
        "checks_total": int,
        "checks": {"vocab_consistency": True, ...},
        "errors": ["check_name: message", ...]
    }
    """
    checks: Dict[str, bool] = {}
    errors: list = []

    if model is not None and tokenizer is not None:
        model.eval()
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map="cpu"
            )
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            return {
                "passed": False,
                "checks_passed": 0,
                "checks_total": 6,
                "checks": {},
                "errors": [f"model_load: {e}"],
            }

    # 1. Vocab consistency
    try:
        assert tokenizer.vocab_size <= model.config.vocab_size, (
            f"tokenizer vocab {tokenizer.vocab_size} > model vocab {model.config.vocab_size}"
        )
        checks["vocab_consistency"] = True
    except Exception as e:
        checks["vocab_consistency"] = False
        errors.append(f"vocab_consistency: {e}")

    # 2. Forward pass valid
    try:
        inputs = tokenizer(TEST_PROMPTS[0], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        assert torch.isfinite(outputs.loss), f"loss is not finite: {outputs.loss.item()}"
        assert torch.all(torch.isfinite(outputs.logits)), "logits contain NaN or Inf"
        checks["forward_pass_valid"] = True
    except Exception as e:
        checks["forward_pass_valid"] = False
        errors.append(f"forward_pass_valid: {e}")

    # 3. Text generation works
    generated_text = ""
    try:
        inputs = tokenizer(TEST_PROMPTS[0], return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=50, do_sample=False
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        assert len(generated_text.strip()) > 0, "generated text is empty"
        checks["text_generation_works"] = True
    except Exception as e:
        checks["text_generation_works"] = False
        errors.append(f"text_generation_works: {e}")

    # 4. Response schema correct
    try:
        assert isinstance(generated_text, str), f"output is {type(generated_text)}, expected str"
        assert len(generated_text) > 0, "output length is 0"
        assert any(c.isalnum() for c in generated_text), "output has no alphanumeric characters"
        checks["response_schema_correct"] = True
    except Exception as e:
        checks["response_schema_correct"] = False
        errors.append(f"response_schema_correct: {e}")

    # 5. Deterministic output
    try:
        inputs = tokenizer(TEST_PROMPTS[1], return_tensors="pt")
        with torch.no_grad():
            out1 = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            out2 = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        assert torch.equal(out1, out2), "two greedy decoding runs produced different outputs"
        checks["deterministic_output"] = True
    except Exception as e:
        checks["deterministic_output"] = False
        errors.append(f"deterministic_output: {e}")

    # 6. No degenerate repetition
    try:
        inputs = tokenizer(TEST_PROMPTS[2], return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=50, do_sample=False
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        unique_tokens = set(new_tokens.tolist())
        assert len(unique_tokens) > 1, (
            f"only {len(unique_tokens)} unique token(s) in generated output"
        )
        checks["no_degenerate_repetition"] = True
    except Exception as e:
        checks["no_degenerate_repetition"] = False
        errors.append(f"no_degenerate_repetition: {e}")

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    checks_passed = sum(1 for v in checks.values() if v)
    return {
        "passed": checks_passed == len(checks),
        "checks_passed": checks_passed,
        "checks_total": len(checks),
        "checks": checks,
        "errors": errors,
    }
