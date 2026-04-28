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
````markdown
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
```

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

The duplicate report is saved in:

`transformed_data/duplicate_rows_with_line_numbers.csv`

This file contains the duplicate rows along with their original CSV row numbers.
