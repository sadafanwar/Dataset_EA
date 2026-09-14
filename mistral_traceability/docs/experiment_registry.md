# Mistral Traceability Experiment Registry

| Experiment ID | Condition | Status | Start Date | End Date | Git Commit | Main Result |
|---|---|---|---|---|---|---|
| M-E1 | Zero-shot baseline | VERIFIED | 2026-09-14 | 2026-09-14 | | Acc=0.4054; Trace R=1.0000; NoTrace R=0.0044; valid=0.9921 |
| M-E2 | 5-shot prompting | PLANNED | | | | |
| M-E3 | 10-shot prompting | PLANNED | | | | |
| M-E4 | QLoRA - original distribution | PLANNED | | | | |
| M-E5 | QLoRA - oversampling | PLANNED | | | | |
| M-E6 | QLoRA - undersampling | PLANNED | | | | |

## Status Definitions

- PLANNED
- PREFLIGHT
- RUNNING
- COMPLETE
- VERIFIED
- DOCUMENTED
- FAILED

## Tracking Rule

No experiment should be executed until:
1. Its configuration file exists.
2. Dataset/fold provenance is recorded.
3. Model identity and revision are recorded.
4. Git commit is recorded.
5. The experiment has an entry in this registry.
