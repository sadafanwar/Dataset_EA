#!/usr/bin/env python3
"""Audit EXP-17 prompt and target token lengths before training."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.exp17.exp17_data import (  # noqa: E402
    build_prompt,
    load_jsonl,
    validate_raw_record,
)

from scripts.exp17.train_exp17 import (  # noqa: E402
    CONFIG_PATH,
    load_yaml,
)


FROZEN_DATASET = (
    REPO_ROOT
    / "data"
    / "traceability_dataset"
    / "traceability_pairs_with_rationale.jsonl"
)

AUDIT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "exp17_qwen35_9b_rationale_single_template"
    / "dataset_token_audit.json"
)


def percentile(
    values: list[int],
    percent: float,
) -> int:
    """Return nearest-rank percentile."""

    if not values:
        raise ValueError(
            "Cannot calculate percentile of empty values."
        )

    ordered = sorted(values)

    rank = math.ceil(
        percent / 100 * len(ordered)
    )

    rank = max(
        1,
        min(rank, len(ordered)),
    )

    return ordered[rank - 1]


def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write("\n")


def main() -> int:
    print()
    print("=" * 70)
    print("EXP-17 COMPLETE TOKEN-LENGTH AUDIT")
    print("=" * 70)

    if not FROZEN_DATASET.is_file():
        raise FileNotFoundError(
            f"Frozen dataset not found: {FROZEN_DATASET}"
        )

    config = load_yaml(CONFIG_PATH)

    model_path = config["model_path"]

    max_length = int(
        config["data"].get(
            "max_length",
            512,
        )
    )

    print(f"Frozen dataset : {FROZEN_DATASET}")
    print(f"Tokenizer      : {model_path}")
    print(f"Max length     : {max_length}")

    print()
    print("Loading tokenizer only...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    records = load_jsonl(FROZEN_DATASET)

    prompt_lengths: list[int] = []
    target_lengths: list[int] = []
    total_lengths: list[int] = []

    truncated_records: list[dict[str, Any]] = []
    truncated_targets: list[dict[str, Any]] = []
    fully_removed_targets: list[dict[str, Any]] = []

    bos_tokens = (
        1
        if tokenizer.bos_token_id is not None
        else 0
    )

    eos_tokens = (
        1
        if tokenizer.eos_token_id is not None
        else 0
    )

    for raw_record in records:
        record = validate_raw_record(
            raw_record
        )

        prompt = build_prompt(
            raw_record,
            include_answer=False,
        )

        target = (
            f"\nLabel: {record['label']}\n"
            f"Rationale: {record['rationale']}"
        )

        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        target_ids = tokenizer.encode(
            target,
            add_special_tokens=False,
        )

        prompt_length = (
            bos_tokens
            + len(prompt_ids)
        )

        target_length = (
            len(target_ids)
            + eos_tokens
        )

        total_length = (
            prompt_length
            + target_length
        )

        prompt_lengths.append(
            prompt_length
        )

        target_lengths.append(
            target_length
        )

        total_lengths.append(
            total_length
        )

        if total_length > max_length:
            retained_target_tokens = max(
                0,
                max_length - prompt_length,
            )

            entry = {
                "pair_id": record["pair_id"],
                "label": record["label"],
                "prompt_tokens": prompt_length,
                "target_tokens": target_length,
                "total_tokens": total_length,
                "retained_target_tokens": (
                    retained_target_tokens
                ),
                "removed_target_tokens": max(
                    0,
                    target_length
                    - retained_target_tokens,
                ),
            }

            truncated_records.append(entry)

            if retained_target_tokens < target_length:
                truncated_targets.append(entry)

            if retained_target_tokens == 0:
                fully_removed_targets.append(
                    entry
                )

    labels = {
        "trace": sum(
            1
            for record in records
            if record["label"] == "trace"
        ),
        "no_trace": sum(
            1
            for record in records
            if record["label"] == "no_trace"
        ),
    }

    report = {
        "experiment_id": "EXP-17",
        "dataset": str(FROZEN_DATASET),
        "model_path": model_path,
        "configured_max_length": max_length,
        "records": len(records),
        "labels": labels,
        "prompt_tokens": {
            "minimum": min(prompt_lengths),
            "mean": mean(prompt_lengths),
            "median": median(prompt_lengths),
            "p90": percentile(
                prompt_lengths,
                90,
            ),
            "p95": percentile(
                prompt_lengths,
                95,
            ),
            "p99": percentile(
                prompt_lengths,
                99,
            ),
            "maximum": max(prompt_lengths),
        },
        "target_tokens": {
            "minimum": min(target_lengths),
            "mean": mean(target_lengths),
            "median": median(target_lengths),
            "p90": percentile(
                target_lengths,
                90,
            ),
            "p95": percentile(
                target_lengths,
                95,
            ),
            "p99": percentile(
                target_lengths,
                99,
            ),
            "maximum": max(target_lengths),
        },
        "total_tokens": {
            "minimum": min(total_lengths),
            "mean": mean(total_lengths),
            "median": median(total_lengths),
            "p90": percentile(
                total_lengths,
                90,
            ),
            "p95": percentile(
                total_lengths,
                95,
            ),
            "p99": percentile(
                total_lengths,
                99,
            ),
            "maximum": max(total_lengths),
        },
        "truncation": {
            "records_exceeding_max_length": (
                len(truncated_records)
            ),
            "targets_partially_truncated": (
                len(truncated_targets)
            ),
            "targets_fully_removed": (
                len(fully_removed_targets)
            ),
            "affected_records": (
                truncated_records
            ),
        },
    }

    save_json(
        AUDIT_OUTPUT,
        report,
    )

    print()
    print("Dataset summary:")
    print(f"Records           : {len(records)}")
    print(f"Trace             : {labels['trace']}")
    print(f"No trace          : {labels['no_trace']}")

    print()
    print("Combined prompt + target lengths:")
    print(
        f"Minimum           : "
        f"{report['total_tokens']['minimum']}"
    )
    print(
        f"Mean              : "
        f"{report['total_tokens']['mean']:.2f}"
    )
    print(
        f"Median            : "
        f"{report['total_tokens']['median']}"
    )
    print(
        f"P90               : "
        f"{report['total_tokens']['p90']}"
    )
    print(
        f"P95               : "
        f"{report['total_tokens']['p95']}"
    )
    print(
        f"P99               : "
        f"{report['total_tokens']['p99']}"
    )
    print(
        f"Maximum           : "
        f"{report['total_tokens']['maximum']}"
    )

    print()
    print("Truncation at configured max length:")
    print(
        "Records affected  : "
        f"{len(truncated_records)}"
    )
    print(
        "Targets truncated : "
        f"{len(truncated_targets)}"
    )
    print(
        "Targets removed   : "
        f"{len(fully_removed_targets)}"
    )

    print()
    print(f"Audit report saved: {AUDIT_OUTPUT}")

    if fully_removed_targets:
        print()
        print("=" * 70)
        print("EXP-17 TOKEN AUDIT FAILED")
        print(
            "At least one record loses its complete "
            "supervised target."
        )
        print("=" * 70)

        return 1

    if truncated_targets:
        print()
        print("=" * 70)
        print(
            "EXP-17 TOKEN AUDIT REQUIRES REVIEW"
        )
        print(
            "Some label/rationale targets are "
            "partially truncated."
        )
        print("=" * 70)

        return 2

    print()
    print("=" * 70)
    print("EXP-17 COMPLETE TOKEN AUDIT PASSED")
    print(
        "No supervised label or rationale target "
        "is truncated."
    )
    print("Model weights were not loaded.")
    print("Training was not started.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
