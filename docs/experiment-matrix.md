# Experiment matrix — RoLoRA reproduction and improvement plan

This matrix turns the project from “run experiments” into “fill report slots.” Keep it updated as runs complete or fail.

## Minimum credible reproduction

Goal: reproduce the central RoLoRA claim on one GLUE dataset with RoBERTa-Large: RoLoRA should degrade less than FedAvg-LoRA and FFA-LoRA as client count increases, and Figure-3-style convergence should show stronger 50-client behavior.

| ID | Dataset | Model | Methods | Clients | Rank | Seeds | Status | Evidence |
|---|---|---|---|---:|---:|---|---|---|
| R0 | QNLI | RoBERTa-base | LoRA, FFA-LoRA, RoLoRA | 3 | 4 | 0 | Local pilot done | `results/table1_pilot_*.log` |
| R1 | QNLI | RoBERTa-base | LoRA, FFA-LoRA, RoLoRA | 3 | 4 | 0 | Medium: RoLoRA done; all modes pending | `results/table1_medium_rolora.log` |
| R2 | MNLI or QNLI | RoBERTa-Large | RoLoRA | 3 | 4 | 0 | Pending feasibility probe | TBD |
| R3 | MNLI or QNLI | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 3 | 4 | 0 | Pending | TBD |
| R4 | MNLI or QNLI | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 20 | 4 | 0 | Pending | TBD |
| R5 | MNLI or QNLI | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 50 | 4 | 0,1,2 | Pending | TBD |

If compute is tight, R5 gets the most seeds because it tests the paper’s main client-scaling claim. R3 and R4 can start with one seed.

## Stretch reproduction

| ID | Dataset | Model | Methods | Clients | Rank | Seeds | Status | Evidence |
|---|---|---|---|---:|---:|---|---|---|
| S1 | SST-2 | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 50 | 4 | 0 | Pending | TBD |
| S2 | QQP | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 50 | 4 | 0 | Pending | TBD |
| S3 | RTE | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 50 | 4 | 0 | Pending | TBD |
| S4 | Best available dataset | RoBERTa-Large | RoLoRA | 50 | 2,8 | 0 | Pending | TBD |

Stretch runs should not displace the minimum credible reproduction.

## Improvement grid

The improvement experiments must answer the unified phase-specific thesis. Do not run large grids until the baseline reproduction is credible.

| ID | Question | Variant | Dataset/model | Clients | Seeds | Status | Success signal |
|---|---|---|---|---:|---|---|---|
| I1 | Does better A initialization reduce early instability or variance? | default vs orthogonal-A | Same as R5 if feasible; otherwise local RoBERTa-base/QNLI | 50 preferred | 0,1,2 if feasible | Pending | Faster early convergence, lower variance, or better final acc. |
| I2 | Does data-informed A help beyond orthogonal A? | SVD/PCA-like A if implementable cheaply | Same as I1 | 50 preferred | 0 or 0,1,2 | Pending | Clear gain over default/orthogonal or useful negative result. |
| I3 | Does RoLoRA need asymmetric A/B learning rates? | LR ratio sweep, e.g. 1:1, 1:4, 1:8, 1:16, maybe 4:1 | Same as R5 or cheaper proxy | 50 preferred | 0 first, then replicate winner | Pending | Faster convergence or better final acc without extra comm. |
| I4 | Does active-factor server momentum help? | FedAvg vs server momentum on active factor | Same as R5 or cheaper proxy | 50 preferred | 0 first, then replicate winner | Pending | Better convergence; if worse, diagnose alternation/momentum conflict. |
| I5 | Does server Adam help or destabilize? | FedAvg vs Adam on active factor | Same as I4 | 50 preferred | 0 first | Pending | Positive or negative result with update-norm explanation. |

## Diagnostics to add before serious runs

For every round, log:

- method;
- dataset;
- seed;
- client count;
- round number;
- active phase (`train A` or `train B`);
- train/eval accuracy and loss;
- active-factor update norm;
- optional frozen-factor equality assertion or marker;
- wall-clock time.

These diagnostics make negative results publishable because they let the team explain why an intervention helped or failed.
