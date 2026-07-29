# EXP-18 Server Runbook

## Purpose

This runbook controls deployment and execution of EXP-18 on the GPU server.

EXP-18 must not be deployed or launched while the EXP-17 automation is active.

Server repository:

`/home/sadaf/traceability_experiments/Dataset_EA`

## 1. EXP-17 completion gate

Before changing the server repository, confirm:

- tmux session `exp17_folds_02_10` has finished
- automation state is `completed`
- automation exit code is `0`
- Fold 01-10 tracking status is `evaluation_completed`
- all expected EXP-17 result files exist
- no training or evaluation process is using the GPU
- EXP-17 results and tracking files are preserved

Preserve this directory:

`results/exp17_qwen35_9b_rationale_single_template/`

Do not use `git clean`, `git reset --hard`, or branch switching before inspecting and protecting the EXP-17 result state.

## 2. Local integration gate

The local `exp18-development` branch must contain:

- EXP-18 configuration and five templates
- deterministic template assignment
- target-masked training dataset
- QLoRA training entrypoint
- controlled per-fold runner
- primary evaluator
- five-template robustness evaluator
- sequential 10-fold orchestrator
- aggregation and comparison script
- experiment manifest and documentation
- release validator

After complete local validation, merge `exp18-development` into `main` and push `main`.

## 3. Controlled server update

The server update must preserve all EXP-17 artifacts.

Before updating code:

1. Record the current branch and HEAD.
2. Inspect `git status --short`.
3. Inspect EXP-17 tracking-file changes.
4. Confirm all EXP-17 results are preserved.
5. Fetch and compare the remote state.
6. Update code without deleting result directories.

Exact update commands will be generated after the final EXP-17 server state is known.

## 4. Read-only EXP-18 preflight

Perform these checks before loading the model or starting training:

- expected server branch and commit are verified
- EXP-18 configuration and scripts are present
- virtual environment Python is available
- required Python packages import successfully
- Qwen3.5-9B model cache is available
- CUDA is available
- BF16 support is available
- sufficient free disk space remains
- no conflicting process is using the GPU
- EXP-18 tracking CSV headers are valid
- no unexpected EXP-18 fold result directories exist
- release validator passes
- orchestrator preflight passes

Release validation command:

```bash
.venv/bin/python scripts/exp18/validate_exp18_release.py
```

Primary execution preflight:

```bash
.venv/bin/python scripts/exp18/run_exp18_all_folds.py --preflight
```

Robustness-enabled preflight:

```bash
.venv/bin/python scripts/exp18/run_exp18_all_folds.py --preflight --include-robustness
```

Preflight must not:

- load model weights
- start training
- start evaluation
- create fold result directories
- modify progress or registry CSV files

## 5. Isolated diagnostic

A controlled diagnostic must run before the final ten-fold experiment.

The diagnostic must use a temporary detached Git worktree at the exact final commit. This prevents diagnostic outputs from entering the final EXP-18 result tree.

The diagnostic must verify:

- model and tokenizer loading
- QLoRA initialization
- deterministic multi-template assignment
- target-only loss masking
- a small training subset
- complete validation-split loss calculation
- validation loss and perplexity recording
- best-model adapter creation
- a small primary evaluation
- unchanged final tracking CSV files
- cleanup of temporary diagnostic artifacts

Diagnostic sample limits must never be used during the final experiment.

Exact diagnostic commands will be generated after the server code update and commit verification.

## 6. Final primary 10-fold execution

Launch the final orchestrator inside a persistent tmux session.

Final command:

```bash
.venv/bin/python scripts/exp18/run_exp18_all_folds.py
```

For each fold, the orchestrator performs:

1. QLoRA training.
2. Complete validation-loss evaluation.
3. Best-model adapter selection.
4. Primary `template_01` unseen-test evaluation.
5. Artifact validation.
6. Progress and registry update.
7. Controlled transition to the next fold.

Final execution must be sequential on the single RTX 4090.

Do not use during final execution:

- `--force`
- `--max-train-samples`
- `--max-test-samples`
- parallel fold execution

If a stage fails, the orchestrator must stop rather than silently continue.

A resumed run must inspect existing artifacts and skip only fully completed stages.

## 7. Robustness execution

The five-template robustness evaluation is secondary.

Run it after all ten primary fold evaluations have completed:

```bash
.venv/bin/python scripts/exp18/run_exp18_all_folds.py --include-robustness
```

The resume-safe orchestrator will skip completed primary stages and execute missing robustness stages.

Each unseen test record is evaluated with all five frozen templates.

Robustness outputs include:

- per-template classification metrics
- unanimous valid-prediction rate
- pairwise label-agreement rate
- all-template valid-output rate
- majority-vote accuracy
- record-level template consistency

The primary EXP-17 versus EXP-18 comparison must continue to use the `template_01` results.

## 8. Final aggregation

After EXP-17 and EXP-18 have both completed:

```bash
.venv/bin/python scripts/analysis/aggregate_exp17_exp18.py --include-robustness
```

Final publication analysis must use all ten paired folds.

Do not use `--allow-incomplete` for final reported results.

Expected outputs include:

- `fold_level_metrics.csv`
- `paired_fold_comparison.csv`
- `exp17_summary.json`
- `exp18_summary.json`
- `paired_comparison_summary.json`
- `exp18_robustness_fold_metrics.csv`
- `exp18_robustness_summary.json`
- `comparison_manifest.json`

## 9. Completion criteria

EXP-18 is complete only when:

- Fold 01-10 training is complete
- Fold 01-10 primary evaluation is complete
- every fold has a valid best-model adapter
- all primary predictions and metrics are preserved
- all progress and registry rows are complete
- robustness evaluation is complete when included
- paired ten-fold aggregation succeeds
- expected prediction counts are verified
- final disk usage is recorded
- branch, commit, configuration, model, and result provenance are recorded
- no diagnostic artifact is mixed with final results
