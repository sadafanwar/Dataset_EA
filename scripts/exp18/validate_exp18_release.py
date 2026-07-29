#!/usr/bin/env python3
"""Validate the EXP-18 local release package."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "exp18"
    / "config_qwen35_9b_exp18.yaml"
)

TEMPLATE_PATH = (
    REPO_ROOT
    / "configs"
    / "exp18"
    / "templates_exp18.yaml"
)

MANIFEST_PATH = (
    REPO_ROOT
    / "configs"
    / "exp18"
    / "experiment_manifest.json"
)

RESULT_ROOT = (
    REPO_ROOT
    / "results"
    / "exp18_qwen35_9b_rationale_multi_template"
)

PROGRESS_FILE = (
    RESULT_ROOT
    / "progress"
    / "progress.csv"
)

REGISTRY_FILE = (
    RESULT_ROOT
    / "registry"
    / "run_registry.csv"
)

REQUIRED_FILES = (
    CONFIG_PATH,
    TEMPLATE_PATH,
    MANIFEST_PATH,
    REPO_ROOT / "docs" / "experiments" / "EXP-18.md",
    REPO_ROOT
    / "docs"
    / "experiments"
    / "EXP-18_SERVER_RUNBOOK.md",
    REPO_ROOT / "scripts" / "exp18" / "exp18_data.py",
    REPO_ROOT
    / "scripts"
    / "exp18"
    / "exp18_training_dataset.py",
    REPO_ROOT / "scripts" / "exp18" / "train_exp18.py",
    REPO_ROOT
    / "scripts"
    / "exp18"
    / "run_exp18_fold.py",
    REPO_ROOT
    / "scripts"
    / "exp18"
    / "evaluate_exp18.py",
    REPO_ROOT
    / "scripts"
    / "exp18"
    / "evaluate_exp18_robustness.py",
    REPO_ROOT
    / "scripts"
    / "exp18"
    / "run_exp18_all_folds.py",
    REPO_ROOT
    / "scripts"
    / "analysis"
    / "aggregate_exp17_exp18.py",
    PROGRESS_FILE,
    REGISTRY_FILE,
)

PYTHON_FILES = tuple(
    path
    for path in REQUIRED_FILES
    if path.suffix == ".py"
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8-sig")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return payload


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8-sig")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected YAML object: {path}"
        )

    return payload


def read_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        return (
            list(reader.fieldnames or []),
            list(reader),
        )


def compile_python(path: Path) -> None:
    source = path.read_text(
        encoding="utf-8-sig"
    )

    compile(
        source,
        str(path),
        "exec",
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--allow-results",
        action="store_true",
        help=(
            "Allow populated tracking files and "
            "fold directories after execution begins."
        ),
    )

    args = parser.parse_args()

    missing = [
        path
        for path in REQUIRED_FILES
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            "Missing release files:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    for path in PYTHON_FILES:
        compile_python(path)

    config = load_yaml(CONFIG_PATH)
    templates = load_yaml(TEMPLATE_PATH)
    manifest = load_json(MANIFEST_PATH)

    if (
        config.get("experiment", {}).get("id")
        != "EXP-18"
    ):
        raise RuntimeError(
            "Configuration experiment ID is invalid."
        )

    if manifest.get("experiment_id") != "EXP-18":
        raise RuntimeError(
            "Manifest experiment ID is invalid."
        )

    template_rows = templates.get("templates")

    if not isinstance(template_rows, list):
        raise RuntimeError(
            "Template catalogue is invalid."
        )

    template_ids = [
        row.get("id")
        for row in template_rows
    ]

    if template_ids != [
        "template_01",
        "template_02",
        "template_03",
        "template_04",
        "template_05",
    ]:
        raise RuntimeError(
            "Unexpected template identifiers."
        )

    if (
        manifest.get("templates", {}).get(
            "primary_test_template"
        )
        != "template_01"
    ):
        raise RuntimeError(
            "Primary template is not template_01."
        )

    progress_headers, progress_rows = read_csv(
        PROGRESS_FILE
    )

    registry_headers, registry_rows = read_csv(
        REGISTRY_FILE
    )

    if not progress_headers:
        raise RuntimeError(
            "progress.csv has no headers."
        )

    if not registry_headers:
        raise RuntimeError(
            "run_registry.csv has no headers."
        )

    fold_directories = sorted(
        path
        for path in RESULT_ROOT.glob(
            "fold_[0-9][0-9]"
        )
        if path.is_dir()
    )

    if not args.allow_results:
        if progress_rows:
            raise RuntimeError(
                "progress.csv contains result rows."
            )

        if registry_rows:
            raise RuntimeError(
                "run_registry.csv contains result rows."
            )

        if fold_directories:
            raise RuntimeError(
                "EXP-18 fold result directories exist."
            )

    technical_doc = (
        REPO_ROOT
        / "docs"
        / "experiments"
        / "EXP-18.md"
    ).read_text(encoding="utf-8-sig")

    runbook = (
        REPO_ROOT
        / "docs"
        / "experiments"
        / "EXP-18_SERVER_RUNBOOK.md"
    ).read_text(encoding="utf-8-sig")

    if "## Robustness evaluation" not in technical_doc:
        raise RuntimeError(
            "Technical-document robustness section is missing."
        )

    if "## 9. Completion criteria" not in runbook:
        raise RuntimeError(
            "Server runbook is incomplete."
        )

    from scripts.analysis.aggregate_exp17_exp18 import (
        run_self_test,
    )

    run_self_test()

    print()
    print("=" * 72)
    print("EXP-18 RELEASE VALIDATION PASSED")
    print("=" * 72)
    print(f"Python files compiled: {len(PYTHON_FILES)}")
    print(f"Frozen templates: {len(template_rows)}")
    print("Primary template: template_01")
    print("Tracking headers: valid")
    print("Aggregation self-test: passed")
    print(
        "Result state: "
        + (
            "allowed"
            if args.allow_results
            else "not started"
        )
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())