"""
Evaluator class for comprehensive model evaluation and benchmarking

This module provides a unified evaluation framework for Large Language Models,
supporting perplexity computation, generation metrics, speed benchmarking,
memory profiling, and model comparison.

Example usage:
    >>> from hivemind_toolkit.evaluation import Evaluator
    >>>
    >>> # Single model evaluation
    >>> evaluator = Evaluator(model=model, tokenizer=tokenizer)
    >>> results = evaluator.compute_perplexity(dataset="wikitext")
    >>>
    >>> # Model comparison
    >>> evaluator = Evaluator(
    ...     model=fine_tuned_model,
    ...     tokenizer=tokenizer,
    ...     reference_model=baseline_model
    ... )
    >>> comparison = evaluator.compare_models(
    ...     model_names=["Fine-tuned", "Baseline"]
    ... )
    >>>
    >>> # Comprehensive evaluation
    >>> results = evaluator.evaluate_comprehensive(
    ...     dataset="wikitext",
    ...     benchmark_speed=True,
    ...     profile_memory=True
    ... )

Academic References:
    - Perplexity: Standard LLM evaluation metric (lower is better)
    - BLEU/ROUGE: Text generation quality metrics
    - Inference speed: Practical deployment consideration
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm
import logging
import time
import numpy as np
import json
import os
from itertools import islice
from prefect import task
from prefect.cache_policies import NO_CACHE

from .device_utils import get_model_device, move_inputs_to_device

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluator for LLM model performance, metrics, and benchmarking

    This class provides a unified interface for evaluating language models across
    multiple dimensions including accuracy (perplexity, generation quality),
    efficiency (inference speed), and resource usage (memory footprint).

    Attributes:
        model: Primary model to evaluate
        tokenizer: Tokenizer for text processing
        device: Device to run evaluation on (cuda/cpu)
        reference_model: Optional reference model for comparison

    Methods:
        compute_perplexity: Compute perplexity on text datasets
        compute_generation_metrics: Compute BLEU, ROUGE, METEOR scores
        benchmark_speed: Benchmark inference speed and throughput
        profile_memory: Profile memory usage and model size
        compare_models: Compare multiple models side-by-side
        evaluate_comprehensive: Run all evaluations at once
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        reference_model: Optional[PreTrainedModel] = None,
    ):
        """
        Initialize evaluator

        Args:
            model: Primary model to evaluate
            tokenizer: Tokenizer for text processing
            device: Device to run evaluation on
            reference_model: Optional reference model for comparison
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.reference_model = reference_model

        # Set models to eval mode
        if self.model is not None:
            self.model.eval()
        if self.reference_model is not None:
            self.reference_model.eval()

        print(f"Evaluator initialized on device: {device}")
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Primary model: {total_params:,} parameters")
        if reference_model is not None:
            ref_params = sum(p.numel() for p in reference_model.parameters())
            print(f"  Reference model: {ref_params:,} parameters")


    @task(name="Compute Perplexity", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
    @torch.no_grad()
    def compute_perplexity(
        self,
        model: Optional[PreTrainedModel] = None,
        dataset: Union[str, List[str]] = "wikitext",
        texts: Optional[List[str]] = None,
        n_samples: int = 500,
        max_length: int = 512,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        """
        Compute perplexity on a dataset

        Perplexity measures how well the model predicts the test data.
        Lower perplexity indicates better prediction quality.
        Formula: perplexity = exp(average_cross_entropy_loss)

        Args:
            model: Model to evaluate (uses self.model if None)
            dataset: Dataset name ('wikitext', 'c4') or list of texts
            texts: Direct list of texts to evaluate (overrides dataset)
            n_samples: Number of samples to evaluate
            max_length: Maximum sequence length
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with perplexity metrics:
                - perplexity: Perplexity score
                - loss: Average cross-entropy loss
                - n_samples: Number of samples evaluated
                - n_tokens: Total number of tokens evaluated

        Example:
            >>> evaluator = Evaluator(model=model, tokenizer=tokenizer)
            >>> results = evaluator.compute_perplexity(dataset="wikitext")
            >>> print(f"Perplexity: {results['perplexity']:.2f}")
        """
        # Use provided model or default to self.model
        model = model or self.model
        if model is None:
            raise ValueError("No model provided. Pass model argument or initialize Evaluator with model.")
        if self.tokenizer is None:
            raise ValueError("No tokenizer available. Initialize Evaluator with tokenizer.")

        model.eval()

        # Load evaluation texts
        if texts is not None:
            eval_texts = texts[:n_samples]
        elif isinstance(dataset, list):
            eval_texts = dataset[:n_samples]
        else:
            eval_texts = self._load_eval_dataset(
                dataset_name=dataset,
                n_samples=n_samples,
                max_length=max_length
            )

        print(f"Computing perplexity on {len(eval_texts)} samples...")

        total_loss = 0.0
        total_tokens = 0
        n_evaluated = 0
        
        # Get model device once (cache it for the loop)
        device = get_model_device(model)
        print(f"Model device: {device}")

        for text in tqdm(eval_texts, desc="Computing perplexity"):
            # Skip empty texts
            if not text or not text.strip():
                print("Skipping empty text in perplexity computation")
                continue

            # Tokenize
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                )
            except Exception as e:
                print(f"Failed to tokenize text: {e}")
                continue

            # Skip if tokenization resulted in empty sequence
            if inputs["input_ids"].numel() == 0:
                print("Skipping text that tokenized to empty sequence")
                continue

            # Move to device (already cached above)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            try:
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                # Accumulate
                n_tokens = inputs["input_ids"].numel()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
                n_evaluated += 1

            except Exception as e:
                print(f"Failed to evaluate sample: {e}")
                continue

        if total_tokens == 0:
            raise ValueError("No valid samples evaluated. Check your dataset and model.")

        # Compute metrics
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        results = {
            "perplexity": float(perplexity),
            "loss": float(avg_loss),
            "n_samples": n_evaluated,
            "n_tokens": total_tokens,
        }

        print(f"Perplexity: {perplexity:.2f} (loss: {avg_loss:.4f})")
        print(f"Evaluated {n_evaluated} samples, {total_tokens} tokens")

        return results


    @task(name="Compute Generation Metrics", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
    def compute_generation_metrics(
        self,
        model: Optional[PreTrainedModel] = None,
        test_data: Optional[List[Dict]] = None,
        metrics: Optional[List[str]] = None,
        max_new_tokens: int = 50,
        num_beams: int = 1,
        do_sample: bool = False,
    ) -> Dict[str, float]:
        """
        Compute generation metrics like BLEU, ROUGE, METEOR

        Evaluates the quality of generated text by comparing model outputs
        against reference texts using multiple metrics.

        Args:
            model: Model to evaluate (uses self.model if None)
            test_data: List of dicts with 'prompt' and 'reference' keys
                Example: [
                    {"prompt": "Translate: Hello", "reference": "Bonjour"},
                    {"prompt": "Summarize: ...", "reference": "Summary text"}
                ]
            metrics: List of metrics to compute (default: ["bleu", "rouge"])
                Available: "bleu", "rouge", "meteor", "bertscore"
            max_new_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search (1 = greedy)
            do_sample: Whether to use sampling

        Returns:
            Dictionary of metric scores:
                - bleu: BLEU score (0-100)
                - rouge1/rouge2/rougeL: ROUGE scores
                - meteor: METEOR score (if requested)
                - bertscore_f1: BERTScore F1 (if requested)

        Example:
            >>> test_data = [
            ...     {"prompt": "Translate to French: Hello", "reference": "Bonjour"},
            ...     {"prompt": "Summarize: The cat sat on the mat.", "reference": "Cat on mat"}
            ... ]
            >>> results = evaluator.compute_generation_metrics(
            ...     test_data=test_data,
            ...     metrics=["bleu", "rouge"]
            ... )
        """
        # Use provided model or default to self.model
        model = model or self.model
        if model is None:
            raise ValueError("No model provided.")
        if self.tokenizer is None:
            raise ValueError("No tokenizer available.")
        if test_data is None or len(test_data) == 0:
            raise ValueError("No test data provided. Pass list of {'prompt': ..., 'reference': ...} dicts.")

        # Default metrics
        if metrics is None:
            metrics = ["bleu", "rouge"]

        model.eval()

        print(f"Computing generation metrics on {len(test_data)} samples...")
        print(f"Metrics: {', '.join(metrics)}")

        # Generate predictions
        predictions = []
        references = []

        for item in tqdm(test_data, desc="Generating predictions"):
            prompt = item.get("prompt", "")
            reference = item.get("reference", "")

            if not prompt or not reference:
                print("Skipping item with empty prompt or reference")
                continue

            # Tokenize prompt
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )

                # Move to device (handles multi-GPU device_map="auto")
                device = get_model_device(model)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Decode
                generated_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

                # Remove prompt from generated text (model often includes it)
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()

                predictions.append(generated_text)
                references.append(reference)

            except Exception as e:
                print(f"Failed to generate for prompt: {e}")
                continue

        if len(predictions) == 0:
            raise ValueError("No successful generations. Check your model and test data.")

        print(f"Generated {len(predictions)} predictions")

        # Compute metrics
        results = {}

        # Import metric libraries as needed
        try:
            # BLEU
            if "bleu" in metrics:
                try:
                    from evaluate import load
                    bleu_metric = load("bleu")
                    bleu_result = bleu_metric.compute(
                        predictions=predictions,
                        references=[[ref] for ref in references]  # List of lists for multi-reference
                    )
                    results["bleu"] = bleu_result["bleu"] * 100  # Convert to 0-100 scale
                except Exception as e:
                    print(f"Failed to compute BLEU: {e}")
                    results["bleu"] = None

            # ROUGE
            if "rouge" in metrics:
                try:
                    from evaluate import load
                    rouge_metric = load("rouge")
                    rouge_result = rouge_metric.compute(
                        predictions=predictions,
                        references=references
                    )
                    results["rouge1"] = rouge_result["rouge1"] * 100
                    results["rouge2"] = rouge_result["rouge2"] * 100
                    results["rougeL"] = rouge_result["rougeL"] * 100
                except Exception as e:
                    print(f"Failed to compute ROUGE: {e}")
                    results["rouge1"] = None
                    results["rouge2"] = None
                    results["rougeL"] = None

            # METEOR
            if "meteor" in metrics:
                try:
                    from evaluate import load
                    meteor_metric = load("meteor")
                    meteor_result = meteor_metric.compute(
                        predictions=predictions,
                        references=references
                    )
                    results["meteor"] = meteor_result["meteor"] * 100
                except Exception as e:
                    print(f"Failed to compute METEOR: {e}")
                    results["meteor"] = None

            # BERTScore (optional, slower)
            if "bertscore" in metrics:
                try:
                    from evaluate import load
                    bertscore_metric = load("bertscore")
                    bertscore_result = bertscore_metric.compute(
                        predictions=predictions,
                        references=references,
                        lang="en"
                    )
                    # Average F1 score
                    results["bertscore_f1"] = np.mean(bertscore_result["f1"]) * 100
                except Exception as e:
                    print(f"Failed to compute BERTScore: {e}")
                    results["bertscore_f1"] = None

        except ImportError as e:
            print(
                f"Failed to import evaluation metrics: {e}\n"
                "Install with: pip install evaluate rouge-score sacrebleu"
            )
            raise

        # Log results
        print("Generation Metrics:")
        for metric_name, score in results.items():
            if score is not None:
                print(f"  {metric_name}: {score:.2f}")
            else:
                print(f"  {metric_name}: N/A (computation failed)")

        return results


    @task(name="Benchmark Speed", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
    @torch.no_grad()
    def benchmark_speed(
        self,
        model: Optional[PreTrainedModel] = None,
        prompt: str = "The quick brown fox jumps over the lazy dog.",
        num_iterations: int = 50,
        max_new_tokens: int = 50,
        warmup_iterations: int = 10,
        mode: str = "generation",  # "generation" or "forward"
    ) -> Dict[str, float]:
        """
        Benchmark inference speed

        Measures the time it takes for the model to perform inference,
        either forward pass only or full text generation.

        Args:
            model: Model to benchmark (uses self.model if None)
            prompt: Test prompt for generation
            num_iterations: Number of iterations to average
            max_new_tokens: Tokens to generate (generation mode only)
            warmup_iterations: Warmup iterations (excluded from timing)
            mode: 'generation' for text generation, 'forward' for forward pass only

        Returns:
            Dictionary with speed metrics:
                - avg_time_ms: Average time per iteration (milliseconds)
                - throughput_per_sec: Iterations per second
                - total_time_sec: Total time for all iterations
                - tokens_per_sec: Tokens per second (generation mode only)

        Example:
            >>> results = evaluator.benchmark_speed(
            ...     prompt="Hello, how are you?",
            ...     num_iterations=100
            ... )
            >>> print(f"Speed: {results['avg_time_ms']:.2f} ms/iteration")
        """
        # Use provided model or default to self.model
        model = model or self.model
        if model is None:
            raise ValueError("No model provided.")
        if self.tokenizer is None:
            raise ValueError("No tokenizer available.")

        model.eval()

        print(f"Benchmarking speed ({mode} mode)...")
        print(f"  Warmup: {warmup_iterations} iterations")
        print(f"  Benchmark: {num_iterations} iterations")

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Move to device (handles multi-GPU device_map="auto")
        device = get_model_device(model)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if mode == "generation":
            # Benchmark text generation
            # Warmup
            print("Running warmup...")
            for _ in range(warmup_iterations):
                _ = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Synchronize for accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark
            print("Running benchmark...")
            times = []
            total_tokens_generated = 0

            for _ in tqdm(range(num_iterations), desc="Benchmarking generation"):
                start_time = time.time()

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.time()
                times.append(end_time - start_time)

                # Count generated tokens (excluding input)
                total_tokens_generated += (outputs.shape[1] - inputs["input_ids"].shape[1])

            # Compute metrics
            avg_time = np.mean(times)
            throughput = 1.0 / avg_time
            total_time = sum(times)
            tokens_per_sec = total_tokens_generated / total_time

            results = {
                "avg_time_ms": avg_time * 1000,
                "throughput_per_sec": throughput,
                "total_time_sec": total_time,
                "tokens_per_sec": tokens_per_sec,
                "mode": "generation",
            }

        elif mode == "forward":
            # Benchmark forward pass only
            # Warmup
            print("Running warmup...")
            for _ in range(warmup_iterations):
                _ = model(**inputs)

            # Synchronize
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark
            print("Running benchmark...")
            times = []

            for _ in tqdm(range(num_iterations), desc="Benchmarking forward pass"):
                start_time = time.time()

                _ = model(**inputs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.time()
                times.append(end_time - start_time)

            # Compute metrics
            avg_time = np.mean(times)
            throughput = 1.0 / avg_time
            total_time = sum(times)

            results = {
                "avg_time_ms": avg_time * 1000,
                "throughput_per_sec": throughput,
                "total_time_sec": total_time,
                "mode": "forward",
            }

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'generation' or 'forward'.")

        # Log results
        print("Speed Benchmark Results:")
        print(f"  Average time: {results['avg_time_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_per_sec']:.2f} iterations/sec")
        if mode == "generation":
            print(f"  Tokens/sec: {results['tokens_per_sec']:.2f}")

        return results


    @task(name="Profile Memory", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
    def profile_memory(
        self,
        model: Optional[PreTrainedModel] = None,
        batch_size: int = 4,
        sequence_length: int = 512,
    ) -> Dict[str, float]:
        """
        Profile memory usage

        Measures model size, parameter count, and GPU memory usage.

        Args:
            model: Model to profile (uses self.model if None)
            batch_size: Batch size for memory profiling
            sequence_length: Sequence length for memory profiling

        Returns:
            Dictionary with memory usage stats:
                - total_params: Total parameters
                - trainable_params: Trainable parameters
                - param_size_mb: Parameter size in MB
                - param_size_gb: Parameter size in GB
                - gpu_memory_allocated_mb: GPU memory allocated (if CUDA)
                - gpu_memory_reserved_mb: GPU memory reserved (if CUDA)

        Example:
            >>> results = evaluator.profile_memory()
            >>> print(f"Model size: {results['param_size_gb']:.2f} GB")
        """
        # Use provided model or default to self.model
        model = model or self.model
        if model is None:
            raise ValueError("No model provided.")

        print("Profiling memory usage...")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Compute parameter size
        param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size_bytes = param_size_bytes + buffer_size_bytes

        param_size_mb = total_size_bytes / (1024 ** 2)
        param_size_gb = total_size_bytes / (1024 ** 3)

        results = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_params_millions": total_params / 1e6,
            "trainable_params_millions": trainable_params / 1e6,
            "param_size_mb": param_size_mb,
            "param_size_gb": param_size_gb,
        }

        # GPU memory usage (if CUDA)
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)

            results["gpu_memory_allocated_mb"] = gpu_memory_allocated
            results["gpu_memory_reserved_mb"] = gpu_memory_reserved
            results["gpu_memory_allocated_gb"] = gpu_memory_allocated / 1024
            results["gpu_memory_reserved_gb"] = gpu_memory_reserved / 1024

        # Log results
        print("Memory Profile:")
        print(f"  Total params: {total_params:,} ({results['total_params_millions']:.1f}M)")
        print(f"  Trainable params: {trainable_params:,} ({results['trainable_params_millions']:.1f}M)")
        print(f"  Parameter size: {param_size_gb:.2f} GB")

        if "gpu_memory_allocated_mb" in results:
            print(f"  GPU memory allocated: {results['gpu_memory_allocated_mb']:.2f} MB")
            print(f"  GPU memory reserved: {results['gpu_memory_reserved_mb']:.2f} MB")

        return results


    @task(name="Compare Models", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
    def compare_models(
        self,
        models: Optional[List[PreTrainedModel]] = None,
        model_names: Optional[List[str]] = None,
        dataset: str = "wikitext",
        n_samples: int = 500,
        benchmark_speed: bool = True,
        profile_memory: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare two or more models side-by-side

        Evaluates multiple models on the same metrics and computes relative
        improvements/degradations.

        Args:
            models: List of models to compare (uses [self.model, self.reference_model] if None)
            model_names: Names for each model (default: ["Model 1", "Model 2", ...])
            dataset: Dataset to evaluate on
            n_samples: Number of samples for evaluation
            benchmark_speed: Whether to benchmark speed
            profile_memory: Whether to profile memory

        Returns:
            Dictionary with comparison results:
                - models: Dict of model_name -> metrics
                - relative_improvements: Improvements of Model 1 vs Model 2
                - summary: Human-readable summary

        Example:
            >>> results = evaluator.compare_models(
            ...     models=[baseline_model, finetuned_model],
            ...     model_names=["Baseline", "Fine-tuned"]
            ... )
            >>> print(results['summary'])
        """
        # Use provided models or default to self.model + self.reference_model
        if models is None:
            if self.model is None or self.reference_model is None:
                raise ValueError(
                    "No models provided. Pass models argument or initialize Evaluator "
                    "with model and reference_model."
                )
            models = [self.reference_model, self.model]  # Compare reference vs primary
            if model_names is None:
                model_names = ["Reference Model", "Primary Model"]

        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(models))]

        if len(models) != len(model_names):
            raise ValueError(f"Number of models ({len(models)}) != number of names ({len(model_names)})")

        print(f"Comparing {len(models)} models: {', '.join(model_names)}")

        comparison_results = {"models": {}}

        # Evaluate each model
        for model, name in zip(models, model_names):
            print(f"\nEvaluating: {name}")
            print("=" * 60)

            model_results = {}

            # Perplexity
            print("Computing perplexity...")
            ppl_results = self.compute_perplexity(
                model=model,
                dataset=dataset,
                n_samples=n_samples
            )
            model_results["perplexity"] = ppl_results

            # Speed
            if benchmark_speed:
                print("Benchmarking speed...")
                speed_results = self.benchmark_speed(
                    model=model,
                    num_iterations=50
                )
                model_results["speed"] = speed_results

            # Memory
            if profile_memory:
                print("Profiling memory...")
                memory_results = self.profile_memory(model=model)
                model_results["memory"] = memory_results

            comparison_results["models"][name] = model_results

        # Compute relative improvements (first model vs second model)
        if len(models) >= 2:
            model1_name = model_names[0]
            model2_name = model_names[1]

            model1_ppl = comparison_results["models"][model1_name]["perplexity"]["perplexity"]
            model2_ppl = comparison_results["models"][model2_name]["perplexity"]["perplexity"]

            ppl_improvement_pct = ((model1_ppl - model2_ppl) / model1_ppl) * 100

            relative_improvements = {
                "perplexity_improvement_pct": ppl_improvement_pct,
                "comparison": f"{model2_name} vs {model1_name}",
            }

            # Speed comparison
            if benchmark_speed:
                model1_time = comparison_results["models"][model1_name]["speed"]["avg_time_ms"]
                model2_time = comparison_results["models"][model2_name]["speed"]["avg_time_ms"]
                speedup = model1_time / model2_time
                relative_improvements["speedup"] = speedup

            # Memory comparison
            if profile_memory:
                model1_size = comparison_results["models"][model1_name]["memory"]["param_size_gb"]
                model2_size = comparison_results["models"][model2_name]["memory"]["param_size_gb"]
                size_ratio = model2_size / model1_size
                memory_reduction_pct = (1 - size_ratio) * 100
                relative_improvements["memory_reduction_pct"] = memory_reduction_pct

            comparison_results["relative_improvements"] = relative_improvements

        # Generate summary
        comparison_results["summary"] = self._generate_comparison_summary(comparison_results)

        # Print summary
        self._print_comparison_results(comparison_results)

        return comparison_results


    @task(name="Comprehensive Evaluation", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
    def evaluate_comprehensive(
        self,
        model: Optional[PreTrainedModel] = None,
        dataset: str = "wikitext",
        texts: Optional[List[str]] = None,
        n_samples: int = 500,
        generation_test_data: Optional[List[Dict]] = None,
        benchmark_speed: bool = True,
        profile_memory: bool = True,
        compute_generation_metrics: bool = False,
        save_results: bool = False,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation (all metrics)

        Convenience method that runs all available evaluation metrics on a model.

        Args:
            model: Model to evaluate (uses self.model if None)
            dataset: Dataset for perplexity evaluation
            texts: Direct list of texts to evaluate (overrides dataset)
            n_samples: Number of samples
            generation_test_data: Test data for generation metrics (optional)
            benchmark_speed: Whether to benchmark speed
            profile_memory: Whether to profile memory
            compute_generation_metrics: Whether to compute generation metrics
            save_results: Whether to save results to file
            output_path: Path to save results (default: "evaluation_results.json")

        Returns:
            Dictionary with all evaluation results

        Example:
            >>> results = evaluator.evaluate_comprehensive(
            ...     dataset="wikitext",
            ...     benchmark_speed=True,
            ...     save_results=True,
            ...     output_path="my_eval_results.json"
            ... )
        """
        print("Running comprehensive evaluation...")
        print("=" * 70)

        results = {}

        # 1. Perplexity
        print("\n[1/4] Computing perplexity...")
        ppl_results = self.compute_perplexity(
            model=model,
            dataset=dataset,
            texts=texts,
            n_samples=n_samples
        )
        results["perplexity"] = ppl_results

        # 2. Generation metrics (optional)
        if compute_generation_metrics and generation_test_data is not None:
            print("\n[2/4] Computing generation metrics...")
            gen_results = self.compute_generation_metrics(
                model=model,
                test_data=generation_test_data
            )
            results["generation_metrics"] = gen_results
        else:
            print("\n[2/4] Skipping generation metrics (not requested or no test data)")

        # 3. Speed benchmark
        if benchmark_speed:
            print("\n[3/4] Benchmarking inference speed...")
            speed_results = self.benchmark_speed(model=model)
            results["speed"] = speed_results
        else:
            print("\n[3/4] Skipping speed benchmark (not requested)")

        # 4. Memory profile
        if profile_memory:
            print("\n[4/4] Profiling memory usage...")
            memory_results = self.profile_memory(model=model)
            results["memory"] = memory_results
        else:
            print("\n[4/4] Skipping memory profile (not requested)")

        # Save results
        if save_results:
            output_path = output_path or "evaluation_results.json"
            self._save_results(results, output_path)
            print(f"Results saved to: {output_path}")

        print("\n" + "=" * 70)
        print("Comprehensive evaluation complete!")
        print("=" * 70)

        return results

    # ==================== Helper Methods ====================


    @task(name="Load Evaluation Dataset", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
    def _load_eval_dataset(
        self,
        dataset_name: str,
        n_samples: int = 500,
        max_length: int = 512,
    ) -> List[str]:
        """
        Load evaluation dataset

        Args:
            dataset_name: Name of dataset ('wikitext', 'c4', etc.)
            n_samples: Number of samples to load
            max_length: Maximum text length (characters)

        Returns:
            List of text strings
        """
        print(f"Loading evaluation dataset: {dataset_name}")

        texts = []

        if dataset_name == "wikitext":
            # Load WikiText-2
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

            for sample in dataset:
                text = sample.get("text", "")

                # Skip empty or very short texts
                if not text or len(text.strip()) < 50:
                    continue

                texts.append(text[:max_length * 10])  # Rough character limit

                if len(texts) >= n_samples:
                    break

        elif dataset_name == "c4":
            # Load C4 validation set (streaming)
            dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

            for sample in islice(dataset, n_samples * 2):  # Load extra in case of filtering
                text = sample.get("text", "")

                # Skip empty or very short texts
                if not text or len(text.strip()) < 50:
                    continue

                texts.append(text[:max_length * 10])

                if len(texts) >= n_samples:
                    break

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'wikitext', 'c4'")

        print(f"Loaded {len(texts)} texts from {dataset_name}")

        return texts


    @task(name="Generate Comparison Summary", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
    def _generate_comparison_summary(self, comparison_results: Dict[str, Any]) -> str:
        """Generate human-readable comparison summary"""
        lines = []
        lines.append("=" * 70)
        lines.append("MODEL COMPARISON SUMMARY")
        lines.append("=" * 70)

        # Model results
        for model_name, metrics in comparison_results["models"].items():
            lines.append(f"\n{model_name}:")

            if "perplexity" in metrics:
                ppl = metrics["perplexity"]["perplexity"]
                lines.append(f"  Perplexity: {ppl:.2f}")

            if "speed" in metrics:
                time_ms = metrics["speed"]["avg_time_ms"]
                lines.append(f"  Speed: {time_ms:.2f} ms/iteration")

            if "memory" in metrics:
                size_gb = metrics["memory"]["param_size_gb"]
                params_m = metrics["memory"]["total_params_millions"]
                lines.append(f"  Size: {size_gb:.2f} GB ({params_m:.1f}M params)")

        # Relative improvements
        if "relative_improvements" in comparison_results:
            imp = comparison_results["relative_improvements"]
            lines.append("\n" + "-" * 70)
            lines.append("RELATIVE IMPROVEMENTS:")
            lines.append(f"  Comparison: {imp['comparison']}")

            ppl_imp = imp.get("perplexity_improvement_pct", 0)
            if ppl_imp > 0:
                lines.append(f"  Perplexity: {ppl_imp:.1f}% improvement [PASS]")
            else:
                lines.append(f"  Perplexity: {abs(ppl_imp):.1f}% degradation [FAIL]")

            if "speedup" in imp:
                speedup = imp["speedup"]
                lines.append(f"  Speedup: {speedup:.2f}x")

            if "memory_reduction_pct" in imp:
                mem_red = imp["memory_reduction_pct"]
                if mem_red > 0:
                    lines.append(f"  Memory: {mem_red:.1f}% reduction")
                else:
                    lines.append(f"  Memory: {abs(mem_red):.1f}% increase")

        lines.append("=" * 70)

        return "\n".join(lines)


    @task(name="Print Comparison Results", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
    def _print_comparison_results(self, comparison_results: Dict[str, Any]):
        """Print comparison results"""
        if "summary" in comparison_results:
            try:
                print("\n" + comparison_results["summary"])
            except UnicodeEncodeError:
                # Fallback for Windows terminals with limited encoding
                summary = comparison_results["summary"]
                # Replace Unicode symbols with ASCII equivalents
                summary = summary.replace("[PASS]", "[+]").replace("[FAIL]", "[-]")
                print("\n" + summary)


    @task(name="Save Results", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
    def _save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file"""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(results)

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to: {output_path}")
