from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-5.5-2026-04-23"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "transformed_data"
    / "cross_dataset_pattern1.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "traceability_dataset"
    / "traceability_pairs_with_rationale.jsonl"
)
DEFAULT_FAILURES = (
    REPO_ROOT
    / "data"
    / "traceability_dataset"
    / "rationale_generation_failures.jsonl"
)
DEFAULT_PROGRESS = (
    REPO_ROOT
    / "data"
    / "traceability_dataset"
    / "rationale_generation_progress.json"
)

EXPECTED_TOTAL = 1887
EXPECTED_COUNTS = {
    "trace": 757,
    "no_trace": 1130,
}

SYSTEM_PROMPT = """
You are generating gold rationales for a software requirements
traceability research dataset.

You will receive:
1. A higher-level requirement.
2. A lower-level requirement.
3. A verified gold label: trace or no_trace.

Your task is not to predict, question, or change the label.
Explain why the supplied gold label is justified.

For label "trace":
Explain the concrete semantic relationship showing how the
lower-level requirement implements, refines, operationalizes,
supports, constrains, or directly contributes to the higher-level
requirement.

For label "no_trace":
Explain why the two requirements do not establish a direct
traceability relationship. Identify the decisive mismatch in
function, objective, actor, component, condition, scope, or expected
behaviour.

Rules:
- Base the rationale only on the supplied requirement texts.
- Do not invent system details, actors, components, or dependencies.
- Do not merely paraphrase the requirements.
- Do not rely only on shared keywords.
- State the decisive semantic reason.
- Write 2 to 4 concise sentences.
- Use professional technical English.
- Return only the requested structured output.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rationales for the traceability dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Source JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Generated rationale JSONL file.",
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=DEFAULT_FAILURES,
        help="Failure log JSONL file.",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=DEFAULT_PROGRESS,
        help="Progress summary JSON file.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum attempts per record.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=1.0,
        help="Base retry delay in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional temporary processing limit.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                raise ValueError(
                    f"Blank line found in input at line {line_number}."
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            validate_source_record(record, line_number)
            records.append(record)

    return records


def validate_source_record(
    record: Any,
    line_number: int,
) -> None:
    if not isinstance(record, dict):
        raise ValueError(
            f"Line {line_number}: record must be a JSON object."
        )

    input_data = record.get("input")

    if not isinstance(input_data, dict):
        raise ValueError(
            f"Line {line_number}: 'input' must be an object."
        )

    for field in (
        "higher_level_requirement",
        "lower_level_requirement",
    ):
        value = input_data.get(field)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Line {line_number}: '{field}' must be a "
                "non-empty string."
            )

    label = record.get("output")

    if label not in {"trace", "no_trace"}:
        raise ValueError(
            f"Line {line_number}: invalid label {label!r}."
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_completed_records(
    output_path: Path,
) -> dict[str, dict[str, Any]]:
    if not output_path.exists():
        return {}

    completed: dict[str, dict[str, Any]] = {}

    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                raise ValueError(
                    f"Blank line in existing output at "
                    f"line {line_number}."
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in existing output at "
                    f"line {line_number}: {exc}"
                ) from exc

            pair_id = record.get("pair_id")

            if not isinstance(pair_id, str):
                raise ValueError(
                    f"Existing output line {line_number} has "
                    "no valid pair_id."
                )

            if pair_id in completed:
                raise ValueError(
                    f"Duplicate pair_id in existing output: {pair_id}"
                )

            completed[pair_id] = record

    return completed


def build_user_prompt(
    higher_requirement: str,
    lower_requirement: str,
    gold_label: str,
) -> str:
    return f"""
Higher-level requirement:
{higher_requirement}

Lower-level requirement:
{lower_requirement}

Verified gold label:
{gold_label}

Generate the rationale according to the instructions.
""".strip()


def request_rationale(
    client: OpenAI,
    higher_requirement: str,
    lower_requirement: str,
    gold_label: str,
) -> tuple[str, str]:
    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=build_user_prompt(
            higher_requirement=higher_requirement,
            lower_requirement=lower_requirement,
            gold_label=gold_label,
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "traceability_rationale",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "rationale": {
                            "type": "string",
                            "description": (
                                "A concise 2-to-4 sentence rationale "
                                "supporting the supplied gold label."
                            ),
                        }
                    },
                    "required": ["rationale"],
                    "additionalProperties": False,
                },
            }
        },
        max_output_tokens=2000,
        store=False,
    )

    if response.status != "completed":
        raise RuntimeError(
            f"Response did not complete. Status: {response.status}"
        )

    if not response.output_text:
        raise RuntimeError("The API returned no output text.")

    try:
        parsed = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The API returned invalid structured JSON."
        ) from exc

    rationale = parsed.get("rationale")

    if not isinstance(rationale, str) or not rationale.strip():
        raise RuntimeError("The API returned an empty rationale.")

    rationale = rationale.strip()
    validate_rationale(rationale)

    return rationale, response.id


def validate_rationale(rationale: str) -> None:
    words = rationale.split()

    if len(words) < 15:
        raise ValueError(
            f"Rationale too short: {len(words)} words."
        )

    if len(words) > 180:
        raise ValueError(
            f"Rationale too long: {len(words)} words."
        )

    sentence_endings = sum(
        rationale.count(mark) for mark in (".", "!", "?")
    )

    if sentence_endings < 2:
        raise ValueError(
            "Rationale appears to contain fewer than two sentences."
        )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def write_progress(
    progress_path: Path,
    *,
    input_path: Path,
    output_path: Path,
    input_hash: str,
    total_source: int,
    completed: int,
    failed: int,
    status: str,
) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": status,
        "model": MODEL,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "source_sha256": input_hash,
        "total_source_records": total_source,
        "completed_records": completed,
        "failed_records": failed,
        "remaining_records": max(total_source - completed, 0),
    }

    temporary_path = progress_path.with_suffix(
        progress_path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(progress_path)
def normalize_output_order(output_path: Path) -> None:
    records = read_generated_output(output_path)

    records.sort(
        key=lambda record: record["source_record_number"]
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

        handle.flush()
        os.fsync(handle.fileno())

    temporary_path.replace(output_path)

def validate_final_output(
    source_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_records = read_generated_output(output_path)

    if len(output_records) != len(source_records):
        raise ValueError(
            "Final output count does not match source count: "
            f"{len(output_records)} != {len(source_records)}"
        )

    labels = Counter(record["label"] for record in output_records)

    if len(source_records) == EXPECTED_TOTAL:
        if labels != Counter(EXPECTED_COUNTS):
            raise ValueError(
                "Final label distribution does not match expected "
                f"counts: {dict(labels)}"
            )

    for index, (source, generated) in enumerate(
        zip(source_records, output_records, strict=True),
        start=1,
    ):
        expected_pair_id = f"PAIR_{index:06d}"
        source_input = source["input"]

        if generated["pair_id"] != expected_pair_id:
            raise ValueError(
                f"Unexpected pair ID at record {index}: "
                f"{generated['pair_id']}"
            )

        if (
            generated["higher_level_requirement"]
            != source_input["higher_level_requirement"]
        ):
            raise ValueError(
                f"Higher-level requirement mismatch at record {index}."
            )

        if (
            generated["lower_level_requirement"]
            != source_input["lower_level_requirement"]
        ):
            raise ValueError(
                f"Lower-level requirement mismatch at record {index}."
            )

        if generated["label"] != source["output"]:
            raise ValueError(
                f"Label mismatch at record {index}."
            )

        validate_rationale(generated["rationale"])


def read_generated_output(
    output_path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    expected_fields = {
        "pair_id",
        "source_record_number",
        "higher_level_requirement",
        "lower_level_requirement",
        "label",
        "rationale",
        "rationale_model",
        "response_id",
    }

    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                raise ValueError(
                    f"Blank output line at {line_number}."
                )

            record = json.loads(line)

            if set(record) != expected_fields:
                raise ValueError(
                    f"Unexpected fields at output line {line_number}: "
                    f"{sorted(record)}"
                )

            records.append(record)

    pair_ids = [record["pair_id"] for record in records]

    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Duplicate pair IDs found in output.")

    return records


def main() -> None:
    args = parse_args()

    load_dotenv(REPO_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY was not found in the .env file."
        )

    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1.")

    source_records = read_jsonl(args.input)
    source_hash = file_sha256(args.input)
    source_counts = Counter(
        record["output"] for record in source_records
    )

    print(f"Input: {args.input}")
    print(f"Records: {len(source_records)}")
    print(f"Labels: {dict(source_counts)}")
    print(f"Source SHA256: {source_hash}")
    print(f"Model: {MODEL}")
    print()

    if len(source_records) == EXPECTED_TOTAL:
        if source_counts != Counter(EXPECTED_COUNTS):
            raise ValueError(
                "Source label distribution differs from the "
                f"expected counts: {dict(source_counts)}"
            )

    completed_records = load_completed_records(args.output)
    completed_count = len(completed_records)
    failure_count = 0

    if completed_count:
        print(
            f"Resume mode: {completed_count} records already completed."
        )

    client = OpenAI()

    source_to_process = source_records

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be greater than zero.")
        source_to_process = source_records[: args.limit]

    write_progress(
        args.progress,
        input_path=args.input,
        output_path=args.output,
        input_hash=source_hash,
        total_source=len(source_records),
        completed=completed_count,
        failed=failure_count,
        status="running",
    )

    for source_index, source_record in enumerate(
        source_to_process,
        start=1,
    ):
        pair_id = f"PAIR_{source_index:06d}"

        if pair_id in completed_records:
            continue

        input_data = source_record["input"]
        higher_requirement = (
            input_data["higher_level_requirement"]
        )
        lower_requirement = (
            input_data["lower_level_requirement"]
        )
        label = source_record["output"]

        print(
            f"[{source_index}/{len(source_records)}] "
            f"{pair_id} ({label})"
        )

        last_error: Exception | None = None

        for attempt in range(1, args.max_retries + 1):
            try:
                rationale, response_id = request_rationale(
                    client=client,
                    higher_requirement=higher_requirement,
                    lower_requirement=lower_requirement,
                    gold_label=label,
                )

                output_record = {
                    "pair_id": pair_id,
                    "source_record_number": source_index,
                    "higher_level_requirement": (
                        higher_requirement
                    ),
                    "lower_level_requirement": (
                        lower_requirement
                    ),
                    "label": label,
                    "rationale": rationale,
                    "rationale_model": MODEL,
                    "response_id": response_id,
                }

                append_jsonl(args.output, output_record)
                completed_records[pair_id] = output_record
                completed_count += 1

                write_progress(
                    args.progress,
                    input_path=args.input,
                    output_path=args.output,
                    input_hash=source_hash,
                    total_source=len(source_records),
                    completed=completed_count,
                    failed=failure_count,
                    status="running",
                )

                break

            except Exception as exc:
                last_error = exc

                if attempt == args.max_retries:
                    failure_record = {
                        "pair_id": pair_id,
                        "source_record_number": source_index,
                        "label": label,
                        "attempts": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    append_jsonl(args.failures, failure_record)
                    failure_count += 1

                    write_progress(
                        args.progress,
                        input_path=args.input,
                        output_path=args.output,
                        input_hash=source_hash,
                        total_source=len(source_records),
                        completed=completed_count,
                        failed=failure_count,
                        status="running_with_failures",
                    )

                    print(
                        f"FAILED after {attempt} attempts: {exc}"
                    )
                    break

                delay = (
                    args.start_delay
                    * (2 ** (attempt - 1))
                    + random.uniform(0.0, 0.5)
                )

                print(
                    f"Attempt {attempt} failed: {exc}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        if last_error is not None and pair_id not in completed_records:
            continue

    is_full_run = args.limit is None

    if is_full_run and failure_count == 0:
        normalize_output_order(args.output)
        validate_final_output(source_records, args.output)

        write_progress(
            args.progress,
            input_path=args.input,
            output_path=args.output,
            input_hash=source_hash,
            total_source=len(source_records),
            completed=completed_count,
            failed=failure_count,
            status="completed_and_validated",
        )

        print()
        print("Rationale dataset completed and validated.")
        print(f"Output: {args.output}")
        print(f"Records: {completed_count}")
        print(f"Trace: {source_counts['trace']}")
        print(f"No trace: {source_counts['no_trace']}")
        return

    final_status = (
        "partial_limit_run"
        if not is_full_run
        else "incomplete_with_failures"
    )

    write_progress(
        args.progress,
        input_path=args.input,
        output_path=args.output,
        input_hash=source_hash,
        total_source=len(source_records),
        completed=completed_count,
        failed=failure_count,
        status=final_status,
    )

    print()
    print(f"Run finished with status: {final_status}")
    print(f"Completed: {completed_count}")
    print(f"Failures in this run: {failure_count}")
    print(f"Progress file: {args.progress}")


if __name__ == "__main__":
    main()