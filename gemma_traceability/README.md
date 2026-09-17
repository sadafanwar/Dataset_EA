# Google Gemma Requirements Traceability Experiments

This directory isolates all Google Gemma experiment code, frozen data copies,
configuration, logs, and results from the existing Qwen and Mistral work.

Model checkpoint:

- `google/gemma-2-9b-it`
- frozen revision: `11c9b309abf73637e4b6f9a3fa1e92e615547819`

Planned sequence:

1. No-fine-tuning prompting: simple zero-shot, simple 5-shot, and simple
   10-shot.
2. No-fine-tuning prompting: 10-run 5-shot and 10-run 10-shot.
3. QLoRA with the original imbalanced training distribution.
4. QLoRA with random oversampling.
5. QLoRA with random undersampling.

Only the first sequence is prepared at present. Later training experiments must
receive their own configs, scripts, logs, and results below this directory.

