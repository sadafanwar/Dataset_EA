"""EXP-18 deterministic multi-template data formatting utilities."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import yaml

from scripts.exp17.exp17_data import (
    SINGLE_TEMPLATE_INSTRUCTION,
    load_jsonl,
    validate_raw_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

TEMPLATE_PATH = (
    REPO_ROOT
    / "configs"
    / "exp18"
    / "templates_exp18.yaml"
)

DEFAULT_TEMPLATE_SEED = 42
PRIMARY_TEMPLATE_ID = "template_01"
EXPECTED_TEMPLATE_COUNT = 5


@lru_cache(maxsize=8)
def load_template_catalog(
    template_path: str | Path = TEMPLATE_PATH,
) -> dict[str, Any]:
    """Load and validate the frozen EXP-18 template catalogue."""

    path = Path(template_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"EXP-18 template file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            "EXP-18 template configuration must be a YAML object."
        )

    raw_templates = payload.get("templates")

    if not isinstance(raw_templates, list):
        raise ValueError(
            "EXP-18 template configuration requires a templates list."
        )

    if len(raw_templates) != EXPECTED_TEMPLATE_COUNT:
        raise ValueError(
            "EXP-18 requires exactly "
            f"{EXPECTED_TEMPLATE_COUNT} templates; "
            f"found {len(raw_templates)}."
        )

    template_ids: list[str] = []
    instructions: dict[str, str] = {}
    roles: dict[str, str] = {}

    for index, raw_template in enumerate(
        raw_templates,
        start=1,
    ):
        if not isinstance(raw_template, dict):
            raise TypeError(
                f"Template {index} must be a YAML object."
            )

        template_id = str(
            raw_template.get("id", "")
        ).strip()

        instruction = str(
            raw_template.get("instruction", "")
        ).strip()

        role = str(
            raw_template.get("role", "")
        ).strip()

        if not template_id:
            raise ValueError(
                f"Template {index} has no identifier."
            )

        if not instruction:
            raise ValueError(
                f"{template_id}: instruction is empty."
            )

        if template_id in instructions:
            raise ValueError(
                f"Duplicate template identifier: {template_id}"
            )

        template_ids.append(template_id)
        instructions[template_id] = instruction
        roles[template_id] = role

    if len(set(instructions.values())) != len(instructions):
        raise ValueError(
            "EXP-18 template instructions must be unique."
        )

    if PRIMARY_TEMPLATE_ID not in instructions:
        raise ValueError(
            f"Primary template {PRIMARY_TEMPLATE_ID!r} is missing."
        )

    if (
        instructions[PRIMARY_TEMPLATE_ID]
        != SINGLE_TEMPLATE_INSTRUCTION
    ):
        raise ValueError(
            "template_01 must exactly match the "
            "EXP-17 canonical instruction."
        )

    output_contract = payload.get(
        "output_contract",
        {},
    )

    if not isinstance(output_contract, dict):
        raise ValueError(
            "output_contract must be a YAML object."
        )

    return {
        "version": str(
            payload.get("version", "")
        ).strip(),
        "template_ids": tuple(template_ids),
        "instructions": instructions,
        "roles": roles,
        "output_contract": output_contract,
        "path": path,
    }


def assigned_template_id(
    pair_id: str,
    *,
    seed: int = DEFAULT_TEMPLATE_SEED,
    template_path: str | Path = TEMPLATE_PATH,
) -> str:
    """
    Deterministically assign one template to one pair.

    Assignment algorithm:
    SHA256(pair_id + NUL + decimal seed), interpreted as
    an unsigned integer, modulo the number of templates.
    """

    normalized_pair_id = str(pair_id).strip()

    if not normalized_pair_id:
        raise ValueError(
            "pair_id must be non-empty for template assignment."
        )

    catalog = load_template_catalog(template_path)
    template_ids = catalog["template_ids"]

    payload = (
        normalized_pair_id
        + "\0"
        + str(int(seed))
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()
    bucket = int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    ) % len(template_ids)

    return str(template_ids[bucket])


def instruction_for_template(
    template_id: str,
    *,
    template_path: str | Path = TEMPLATE_PATH,
) -> str:
    """Return the frozen instruction for one template ID."""

    catalog = load_template_catalog(template_path)
    instructions = catalog["instructions"]

    if template_id not in instructions:
        raise ValueError(
            f"Unknown EXP-18 template ID: {template_id!r}"
        )

    return str(instructions[template_id])


def build_prompt(
    record: dict[str, Any],
    include_answer: bool,
    *,
    template_id: str | None = None,
    seed: int = DEFAULT_TEMPLATE_SEED,
    template_path: str | Path = TEMPLATE_PATH,
) -> str:
    """
    Build one EXP-18 prompt.

    When template_id is omitted, the template is assigned
    deterministically from pair_id and seed.
    """

    item = validate_raw_record(record)

    selected_template_id = (
        template_id
        if template_id is not None
        else assigned_template_id(
            item["pair_id"],
            seed=seed,
            template_path=template_path,
        )
    )

    instruction = instruction_for_template(
        selected_template_id,
        template_path=template_path,
    )

    prompt = (
        f"{instruction}\n\n"
        "Higher-level requirement:\n"
        f"{item['higher_level_requirement']}\n\n"
        "Lower-level requirement:\n"
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


def build_training_texts(
    records: Iterable[dict[str, Any]],
    *,
    seed: int = DEFAULT_TEMPLATE_SEED,
) -> list[str]:
    """Build target-bearing texts with assigned templates."""

    return [
        build_prompt(
            record,
            include_answer=True,
            seed=seed,
        )
        for record in records
    ]


def build_validation_texts(
    records: Iterable[dict[str, Any]],
    *,
    seed: int = DEFAULT_TEMPLATE_SEED,
) -> list[str]:
    """Build validation texts using the same assignment policy."""

    return build_training_texts(
        records,
        seed=seed,
    )


def build_assigned_inference_prompts(
    records: Iterable[dict[str, Any]],
    *,
    seed: int = DEFAULT_TEMPLATE_SEED,
) -> list[str]:
    """Build answer-free prompts with assigned templates."""

    return [
        build_prompt(
            record,
            include_answer=False,
            seed=seed,
        )
        for record in records
    ]


def build_primary_inference_prompts(
    records: Iterable[dict[str, Any]],
) -> list[str]:
    """
    Build canonical answer-free prompts for primary comparison.

    template_01 exactly matches the EXP-17 test instruction.
    """

    return [
        build_prompt(
            record,
            include_answer=False,
            template_id=PRIMARY_TEMPLATE_ID,
        )
        for record in records
    ]


def build_robustness_prompt_matrix(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build one answer-free test prompt per record and template."""

    catalog = load_template_catalog()
    rows: list[dict[str, str]] = []

    for record in records:
        item = validate_raw_record(record)

        for template_id in catalog["template_ids"]:
            rows.append(
                {
                    "pair_id": item["pair_id"],
                    "template_id": str(template_id),
                    "prompt": build_prompt(
                        record,
                        include_answer=False,
                        template_id=str(template_id),
                    ),
                }
            )

    return rows


def build_assignment_rows(
    records: Iterable[dict[str, Any]],
    *,
    seed: int = DEFAULT_TEMPLATE_SEED,
) -> list[dict[str, str | int]]:
    """Return an auditable pair-to-template assignment table."""

    rows: list[dict[str, str | int]] = []

    for record in records:
        item = validate_raw_record(record)

        rows.append(
            {
                "pair_id": item["pair_id"],
                "template_id": assigned_template_id(
                    item["pair_id"],
                    seed=seed,
                ),
                "seed": int(seed),
                "assignment_method": (
                    "sha256_pair_id_seed_modulo"
                ),
            }
        )

    return rows
