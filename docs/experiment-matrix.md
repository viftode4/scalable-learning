# Experiment matrix — RoLoRA reproduction and improvement plan

This matrix turns the project from "run experiments" into "fill report slots."
Keep it updated as runs complete or fail.

## Dataset rule

- **Primary:** MNLI, because it is the proposal/paper-default target for the
  main reproduction story.
- **Fallback:** QNLI, only when MNLI data prep, runtime, or feasibility blocks
  progress. QNLI is acceptable as a fallback because the supplement pipeline is
  already locally verified there.

## Compute ladder and stop gates

| Gate | Target | Budget cap | Pass condition | Stop / fallback condition |
|---|---|---:|---|---|
| H0 | Tests + MNIST + supplement smoke | laptop minutes | `make check` and smoke markers pass. | Fix local setup before any bigger run. |
| H1 | RoBERTa-base/QNLI pilot | laptop hours | All three modes complete. | Do not spend GPU until runner/config issues are resolved. |
| H2 | RoBERTa-base/QNLI medium | overnight local at most | `table1-medium-all` completes or fails with ledgered reason. | If too slow, keep only RoLoRA plus one baseline and move to feasibility. |
| H3 | RoBERTa-Large feasibility | one short GPU job | Model/data load, one tiny run, marker and metrics/failure captured. | Use QNLI or smaller local diagnostics if memory/runtime blocks. |
| H4 | Selected reproduction | cluster only | 3/20/50-client rows for one dataset. | Stop broadening if 3-client baseline misses paper by >±2%. |
| H5 | Improvement grid | after diagnostics | At least one axis has baseline + curve + phase explanation. | Reframe as negative result if no accuracy gain but diagnostics are clear. |

## Minimum credible reproduction

Goal: reproduce the central RoLoRA claim on one GLUE dataset with
RoBERTa-Large: RoLoRA should degrade less than FedAvg-LoRA and FFA-LoRA as
client count increases, and Figure-3-style convergence should show stronger
50-client behavior.

| ID | Dataset | Model | Methods | Clients | Rank | Seeds | Status | Evidence |
|---|---|---|---|---:|---:|---|---|---|
| R0 | QNLI | RoBERTa-base | LoRA, FFA-LoRA, RoLoRA | 3 | 4 | 0 | Local pilot done | `results/table1_pilot_*.log` |
| R1 | QNLI | RoBERTa-base | LoRA, FFA-LoRA, RoLoRA | 3 | 4 | 0 | Medium: RoLoRA done; all modes pending | `results/table1_medium_rolora.log` |
| R2 | MNLI primary; QNLI fallback | RoBERTa-Large | RoLoRA | 3 | 4 | 0 | Config ready; GPU run pending | `experiments/configs/roberta_large_feasibility.yaml`, `make roberta-large-feasibility MODE=rolora` |
| R3 | MNLI primary; QNLI fallback | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 3 | 4 | 0 | Pending | TBD |
| R4 | MNLI primary; QNLI fallback | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 20 | 4 | 0 | Pending | TBD |
| R5 | MNLI primary; QNLI fallback | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 50 | 4 | 0,1,2 | Pending | TBD |

If compute is tight, R5 gets the most seeds because it tests the paper's main
client-scaling claim. R3 and R4 can start with one seed.

## Stretch reproduction

| ID | Dataset | Model | Methods | Clients | Rank | Seeds | Status | Evidence |
|---|---|---|---|---:|---:|---|---|---|
| S1 | SST-2 | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 50 | 4 | 0 | Pending | TBD |
| S2 | QQP | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 50 | 4 | 0 | Pending | TBD |
| S3 | RTE | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 50 | 4 | 0 | Pending | TBD |
| S4 | Best available dataset | RoBERTa-Large | RoLoRA | 50 | 2,8 | 0 | Pending | TBD |

Stretch runs must not displace the minimum credible reproduction.

## Improvement grid

The improvement experiments must answer the unified phase-specific thesis. Do
not run large grids until baseline reproduction and mandatory diagnostics are
credible.

| ID | Question | Variant | Dataset/model | Clients | Seeds | Status | Success signal |
|---|---|---|---|---:|---|---|---|
| I1 | Does better A initialization reduce early instability or variance? | default vs orthogonal-A | Same as R5 if feasible; otherwise local RoBERTa-base/QNLI | 50 preferred | 0,1,2 if feasible | Pending | Faster early convergence, lower variance, or better final acc. |
| I2 | Does data-informed A help beyond orthogonal A? | SVD/PCA-like A only if cheap and leakage-safe | Same as I1 | 50 preferred | 0 or 0,1,2 | Pending | Clear gain over default/orthogonal or useful negative result. |
| I3 | Does RoLoRA need asymmetric A/B learning rates? | 1:1, paper-near B=2×A and B=4×A, then 8×/16× only if diagnostics justify | Same as R5 or cheaper proxy | 50 preferred | 0 first, then replicate winner | Pending | Faster convergence or better final acc without extra comm. |
| I4 | Does active-factor server momentum help? | FedAvg vs server momentum on active factor | Same as R5 or cheaper proxy | 50 preferred | 0 first, then replicate winner | Pending | Better convergence; if worse, diagnose alternation/momentum conflict. |
| I5 | Does server Adam help or destabilize? | FedAvg vs Adam on active factor | Same as I4 | 50 preferred | 0 first | Pending | Positive or negative result with update-norm explanation. |

## Comparability constraints

- Orthogonal A must be the first initialization variant because it is cheap and
  does not need data access.
- SVD/PCA-like A must document the data source. It may use public/pretrained or
  unlabeled training data only; no test-label leakage.
- A/B LR sweeps must acknowledge the paper's asymmetric LR ablations before
  claiming novelty.
- Server optimizers update the active factor only. The frozen factor must remain
  bit-identical across clients/server. Moment state policy must be recorded as
  either `persist-same-factor` or `reset-on-phase-switch` before the run.

## Mandatory diagnostics before serious runs

For every serious run, log or extract:

- method;
- dataset;
- seed;
- client count;
- config path and git SHA;
- round number;
- active phase (`train A`, `train B`, or `train both`);
- train/eval accuracy and loss when available;
- active-factor update norm;
- frozen-factor equality marker;
- wall-clock time;
- peak memory if feasible;
- failure reason for incomplete runs.

These diagnostics make negative results publishable because they let the team
explain why an intervention helped or failed.
