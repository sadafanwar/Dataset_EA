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