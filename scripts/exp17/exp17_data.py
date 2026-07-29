"""EXP-17 dataset loading and single-template formatting utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


VALID_LABELS = {"trace", "no_trace"}

SINGLE_TEMPLATE_INSTRUCTION = (
    "Determine whether the lower-level requirement is traceable to the "
    "higher-level requirement. Return the label as either 'trace' or "
    "'no_trace', followed by a concise rationale explaining the decision."
)


def _require_text(
    record: dict[str, Any],
    field: str,
    pair_id: str,
) -> str:
    """Return a required non-empty text field."""

    if field not in record:
        raise ValueError(
            f"{pair_id}: missing required field '{field}'"
        )

    value = record[field]

    if value is None:
        raise ValueError(
            f"{pair_id}: field '{field}' is null"
        )

    text = str(value).strip()

    if not text:
        raise ValueError(
            f"{pair_id}: field '{field}' is empty"
        )

    return text


def validate_raw_record(
    record: dict[str, Any],
) -> dict[str, str]:
    """Validate and normalize one frozen EXP-17 record."""

    if not isinstance(record, dict):
        raise TypeError("EXP-17 record must be a JSON object")

    pair_id = _require_text(
        record,
        "pair_id",
        "<unknown>",
    )

    higher_requirement = _require_text(
        record,
        "higher_level_requirement",
        pair_id,
    )

    lower_requirement = _require_text(
        record,
        "lower_level_requirement",
        pair_id,
    )

    label = _require_text(
        record,
        "label",
        pair_id,
    ).lower()

    rationale = _require_text(
        record,
        "rationale",
        pair_id,
    )

    if label not in VALID_LABELS:
        raise ValueError(
            f"{pair_id}: invalid label '{label}'. "
            f"Expected one of {sorted(VALID_LABELS)}"
        )

    return {
        "pair_id": pair_id,
        "higher_level_requirement": higher_requirement,
        "lower_level_requirement": lower_requirement,
        "label": label,
        "rationale": rationale,
    }


def build_prompt(
    record: dict[str, Any],
    include_answer: bool,
) -> str:
    """
    Build the EXP-17 single-template text.

    include_answer=True:
        Used for supervised training and validation loss.

    include_answer=False:
        Used later for test-set generation.
    """

    item = validate_raw_record(record)

    prompt = (
        f"{SINGLE_TEMPLATE_INSTRUCTION}\n\n"
        f"Higher-level requirement:\n"
        f"{item['higher_level_requirement']}\n\n"
        f"Lower-level requirement:\n"
        f"{item['lower_level_requirement']}\n\n"
        "Answer:"
    )

    if not include_answer:
        return prompt

    target = (
        f"Label: {item['label']}\n"
        f"Rationale: {item['rationale']}"
    )

    return f"{prompt}\n{target}"


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Read and validate a JSONL dataset."""

    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    records: list[dict[str, Any]] = []

    with file_path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path} "
                    f"at line {line_number}: {error}"
                ) from error

            validate_raw_record(record)
            records.append(record)

    if not records:
        raise ValueError(
            f"No records found in dataset: {file_path}"
        )

    return records


def build_training_texts(
    records: Iterable[dict[str, Any]],
) -> list[str]:
    """Convert raw records to supervised training texts in memory."""

    return [
        build_prompt(record, include_answer=True)
        for record in records
    ]


def build_inference_prompts(
    records: Iterable[dict[str, Any]],
) -> list[str]:
    """Convert raw records to prompts without gold answers."""

    return [
        build_prompt(record, include_answer=False)
        for record in records
    ]
