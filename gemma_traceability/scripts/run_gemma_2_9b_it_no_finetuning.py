#!/usr/bin/env python3

import argparse
import csv
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from no_finetuning_common import (
    METRIC_KEYS,
    aggregate_metrics,
    audit_datasets,
    build_task_prompt,
    compute_metrics,
    generate_prediction,
    get_label,
    get_requirements,
    gpu_memory_mib,
    load_config,
    load_jsonl,
    load_model,
    resolve_path,
    save_csv,
    save_json,
    set_reproducibility,
)


PREDICTION_FIELDS = [
    "example_id",
    "gold_label",
    "pred_label",
    "raw_prediction",
    "shot_setting",
    "run",
    "seed",
    "prompt_tokens",
    "effective_prompt_limit",
    "prompt_truncated",
    "generated_tokens",
    "generation_seconds",
    "higher_level_requirement",
    "lower_level_requirement",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Gemma 2 9B Instruct no-fine-tuning traceability prompting."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["simple", "multi", "all"], default="all")
    return parser.parse_args()


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_setting(
    *,
    method,
    run,
    seed,
    demo_file,
    test_file,
    prediction_file,
    metrics_file,
    tokenizer,
    model,
    config,
):
    demo_examples = load_jsonl(demo_file)
    test_rows = load_jsonl(test_file)
    prediction_file = Path(prediction_file)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = read_csv(prediction_file)
    existing_by_id = {row["example_id"]: row for row in existing_rows}
    mode = "a" if prediction_file.exists() and prediction_file.stat().st_size else "w"
    print(
        f"Starting {method} run={run}: demos={len(demo_examples)} "
        f"test={len(test_rows)} resume_rows={len(existing_rows)}",
        flush=True,
    )

    with prediction_file.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        if mode == "w":
            writer.writeheader()
        for index, row in enumerate(test_rows, start=1):
            example_id = str(row.get("example_id") or f"line_{row['_line_id']}")
            if example_id in existing_by_id:
                continue
            high, low = get_requirements(row)
            generated = generate_prediction(
                build_task_prompt(demo_examples, row), tokenizer, model, config
            )
            output = {
                "example_id": example_id,
                "gold_label": get_label(row),
                "pred_label": generated["pred_label"],
                "raw_prediction": generated["raw_prediction"],
                "shot_setting": method,
                "run": run,
                "seed": seed,
                "prompt_tokens": generated["prompt_tokens"],
                "effective_prompt_limit": generated["effective_prompt_limit"],
                "prompt_truncated": generated["prompt_truncated"],
                "generated_tokens": generated["generated_tokens"],
                "generation_seconds": generated["generation_seconds"],
                "higher_level_requirement": high,
                "lower_level_requirement": low,
            }
            writer.writerow(output)
            handle.flush()
            existing_by_id[example_id] = {key: str(value) for key, value in output.items()}
            completed = len(existing_by_id)
            if completed % 10 == 0 or completed == len(test_rows):
                print(
                    f"{method} run={run}: evaluated {completed}/{len(test_rows)}",
                    flush=True,
                )

    final_rows = read_csv(prediction_file)
    if len(final_rows) != len(test_rows):
        raise RuntimeError(
            f"Prediction row count mismatch for {method} run={run}: "
            f"expected {len(test_rows)}, observed {len(final_rows)}"
        )
    metrics = compute_metrics(final_rows, method, run, seed)
    metrics.update(
        {
            "prediction_file": str(prediction_file.relative_to(resolve_path("."))).replace("\\", "/"),
            "demo_file": str(Path(demo_file).relative_to(resolve_path("."))).replace("\\", "/"),
            "test_file": str(Path(test_file).relative_to(resolve_path("."))).replace("\\", "/"),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_json(metrics, metrics_file)
    print(f"Completed {method} run={run}: accuracy={metrics['accuracy']:.4f}", flush=True)
    return metrics


def run_simple(config, tokenizer, model, results_dir):
    data_dir = resolve_path(config["data"]["simple_dir"])
    output_dir = results_dir / "simple"
    test_file = data_dir / "test_100.jsonl"
    settings = [
        ("zero_shot", data_dir / "zero_shot_examples.jsonl"),
        ("five_shot", data_dir / "five_shot_examples.jsonl"),
        ("ten_shot", data_dir / "ten_shot_examples.jsonl"),
    ]
    rows = []
    for method, demo_file in settings:
        rows.append(
            run_setting(
                method=method,
                run=1,
                seed=int(config["inference"]["seed"]),
                demo_file=demo_file,
                test_file=test_file,
                prediction_file=output_dir / f"{method}_predictions.csv",
                metrics_file=output_dir / f"{method}_metrics.json",
                tokenizer=tokenizer,
                model=model,
                config=config,
            )
        )
    save_json(rows, output_dir / "simple_summary.json")
    csv_rows = []
    for row in rows:
        item = dict(row)
        item["confusion_matrix"] = json.dumps(item["confusion_matrix"])
        csv_rows.append(item)
    save_csv(csv_rows, output_dir / "simple_summary.csv")
    return rows


def run_multi(config, tokenizer, model, results_dir):
    data_dir = resolve_path(config["data"]["multi_run_dir"])
    output_dir = results_dir / "multi_run"
    test_file = data_dir / "test_100.jsonl"
    rows = []
    for run_index, seed in enumerate(config["data"]["run_seeds"], start=1):
        label = f"run_{run_index:02d}"
        for method in ["five_shot", "ten_shot"]:
            rows.append(
                run_setting(
                    method=method,
                    run=run_index,
                    seed=int(seed),
                    demo_file=data_dir / f"{method}_{label}_examples.jsonl",
                    test_file=test_file,
                    prediction_file=output_dir / f"{method}_{label}_predictions.csv",
                    metrics_file=output_dir / f"{method}_{label}_metrics.json",
                    tokenizer=tokenizer,
                    model=model,
                    config=config,
                )
            )
    save_json(rows, output_dir / "multi_run_summary.json")
    csv_rows = []
    for row in rows:
        item = dict(row)
        item["confusion_matrix"] = json.dumps(item["confusion_matrix"])
        csv_rows.append(item)
    save_csv(csv_rows, output_dir / "multi_run_summary.csv")

    aggregate = [
        aggregate_metrics(rows, "five_shot"),
        aggregate_metrics(rows, "ten_shot"),
    ]
    save_json(aggregate, output_dir / "multi_run_aggregate.json")
    aggregate_csv = []
    for row in aggregate:
        item = dict(row)
        item["confusion_matrix_sum"] = json.dumps(item["confusion_matrix_sum"])
        aggregate_csv.append(item)
    save_csv(aggregate_csv, output_dir / "multi_run_aggregate.csv")
    return rows, aggregate


def load_json(path):
    if not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_comparison(results_dir):
    comparison = []
    simple = load_json(results_dir / "simple" / "simple_summary.json") or []
    for metrics in simple:
        comparison.append(
            {
                "experiment": metrics["method"].replace("_", " ").title(),
                "few_shot": metrics["method"].replace("_", " "),
                "runs": 1,
                **{key: metrics[key] for key in METRIC_KEYS},
                "invalid_predictions": metrics["invalid_predictions"],
            }
        )
    aggregate = load_json(results_dir / "multi_run" / "multi_run_aggregate.json") or []
    for metrics in aggregate:
        comparison.append(
            {
                "experiment": f"Multi-run {metrics['method'].replace('_', ' ')}".title(),
                "few_shot": f"{metrics['method'].replace('_', ' ')} multi-run",
                "runs": metrics["runs"],
                **{key: metrics[f"{key}_mean"] for key in METRIC_KEYS},
                "invalid_predictions": metrics["total_invalid_predictions"],
            }
        )
    if comparison:
        save_csv(comparison, results_dir / "comparison_summary.csv")
        save_json(comparison, results_dir / "comparison_summary.json")


def main():
    args = parse_args()
    config = load_config(args.config)
    results_dir = resolve_path(config["outputs"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = results_dir / f"run_manifest_{args.phase}.json"
    manifest = {
        "experiment": config["experiment"],
        "model": config["model"],
        "inference": config["inference"],
        "phase": args.phase,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "errors": [],
    }
    started = time.perf_counter()
    try:
        print("Auditing frozen prompting data...", flush=True)
        audit = audit_datasets(config, strict=True)
        save_json(audit, results_dir / "data_audit.json")
        manifest["data_audit_status"] = audit["status"]

        set_reproducibility(int(config["inference"]["seed"]))
        manifest["gpu_before_load_mib"] = gpu_memory_mib()
        print("Loading Gemma 2 9B Instruct once for all requested settings...", flush=True)
        tokenizer, model, model_details = load_model(config)
        manifest["model_loaded"] = model_details
        manifest["gpu_after_load_mib"] = gpu_memory_mib()

        if args.phase in {"simple", "all"}:
            run_simple(config, tokenizer, model, results_dir)
        if args.phase in {"multi", "all"}:
            run_multi(config, tokenizer, model, results_dir)
        save_comparison(results_dir)

        manifest["gpu_after_evaluation_mib"] = gpu_memory_mib()
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["errors"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(manifest["errors"][-1]["traceback"], flush=True)
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(manifest, run_manifest_path)
        print(f"Run manifest: {run_manifest_path}", flush=True)
        print(f"Run status: {manifest['status']}", flush=True)

    if manifest["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
