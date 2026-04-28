# Cross-Level Requirement Traceability Dataset Transformation

## Purpose

This repository contains the transformation of a cross-level requirements traceability dataset into Pattern 1 pairwise classification format for LLM fine-tuning experiments.

## Original Dataset Structure

The original CSV contains:

- `high_text`: higher-level requirement
- `low_text`: lower-level requirement
- `label`: binary traceability label

Label meaning:

- `1` = trace link exists
- `0` = no trace link

## Pattern 1 Format

Each row is converted into this format:

```json
{
  "instruction": "Determine whether the lower-level requirement traces to the higher-level requirement.",
  "input": {
    "higher_level_requirement": "...",
    "lower_level_requirement": "..."
  },
  "output": "trace"
}

## Dataset Summary

This dataset is a binary cross-level requirement traceability dataset.

Each row contains:

- one higher-level requirement
- one lower-level requirement
- one binary traceability label

## Dataset Statistics

| Dataset Version | Total Examples | Trace Examples | No-trace Examples | Notes |
|---|---:|---:|---:|---|
| Original CSV | 1,892 | 757 | 1,135 | Raw dataset before preprocessing |
| After duplicate removal | 1,887 | 757 | 1,130 | Final cleaned dataset used for Pattern 1 conversion |

## Label Distribution

Original dataset:

| Label | Meaning | Count |
|---|---|---:|
| `1` | trace link exists | 757 |
| `0` | no trace link | 1,135 |

After duplicate removal:

| Output Label | Meaning | Count |
|---|---|---:|
| `trace` | lower-level requirement traces to higher-level requirement | 757 |
| `no_trace` | no traceability link exists | 1,130 |

## Duplicate Analysis

The original dataset contained 1,892 rows.

During preprocessing, exact duplicate rows were checked using the following fields:

- `high_text`
- `low_text`
- `label`

Five duplicate rows were removed.

All removed duplicates belonged to the `no_trace` class.

Summary:

| Duplicate Check | Count |
|---|---:|
| Original rows | 1,892 |
| Duplicate rows removed | 5 |
| Final rows after duplicate removal | 1,887 |

The duplicate report is saved in:

`transformed_data/duplicate_rows_with_line_numbers.csv`

This file contains the duplicate rows along with their original CSV row numbers.
