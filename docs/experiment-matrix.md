# Experiment matrix — RoLoRA reproduction and improvement plan

This matrix turns the project from "run experiments" into "fill report slots."
Keep it updated as runs complete or fail.

## Dataset rule

- **First sweep:** QNLI. The authors' supplement ships only QNLI data prep
  (`sst2/qnli2json.py`), and the full pipeline is locally verified end-to-end
  on it. Principle: get one dataset reproduced fully and properly before any
  second dataset.
- **Second sweep (expansion):** MNLI. The QNLI loader pattern
  (`datasets.load_dataset("glue", "qnli")` + JSON formatting) extends
  trivially to MNLI; the only real work is writing an `mnli2json.py`
  analogue and confirming the field mapping.
- **Stretch:** the remaining Table-1 tasks (SST-2, QQP, RTE) only after the
  first two sweeps are fully ledgered and the report cells are filled.

## Methods scope and FlexLoRA gap

Paper Table 1 reports four methods: LoRA, FFA-LoRA, FlexLoRA, RoLoRA. The
authors' supplement does **not** ship FlexLoRA code. We reproduce
**LoRA, FFA-LoRA, RoLoRA** via `SLS_ALTERNATION_MODE`. FlexLoRA is omitted and
the omission must be disclosed explicitly in the report's reproduction section
("FlexLoRA results from Table 1 are not reproduced; the method is not included
in the authors' supplementary material and we did not re-implement it").

## Compute ladder and stop gates

| Gate | Target | Budget cap | Pass condition | Stop / fallback condition |
|---|---|---:|---|---|
| H0 | Tests + MNIST + supplement smoke | laptop minutes | `make check` and smoke markers pass. | Fix local setup before any bigger run. |
| H1 | RoBERTa-base/QNLI pilot | laptop hours | All three modes complete. | Do not spend GPU until runner/config issues are resolved. |
| H2 | RoBERTa-base/QNLI medium | overnight local at most | `table1-medium-all` completes or fails with ledgered reason. | If too slow, keep only RoLoRA plus one baseline and move to feasibility. |
| H3 | RoBERTa-Large feasibility | one short GPU job | Model/data load, one tiny run, marker and metrics/failure captured. | Use QNLI or smaller local diagnostics if memory/runtime blocks. |
| H4 | QNLI reproduction sweep | cluster only | All 4 cells × 3 methods × 3 seeds (36 jobs) complete on QNLI. | Stop broadening if 3-client baseline misses paper by >±2% without an explainable setup cause. |
| H5 | MNLI reproduction (expansion) | cluster only | Same 4×3×3 shape on MNLI once H4 is fully ledgered. | Skip if H4 used all available cluster budget. |
| H6 | Improvement grid | after diagnostics | At least one improvement axis has baseline + curve + phase explanation. | Reframe as negative result if no accuracy gain but diagnostics are clear. |

## Minimum credible reproduction

Goal: reproduce the central RoLoRA claim on one GLUE dataset with
RoBERTa-Large: RoLoRA should degrade less than FedAvg-LoRA and FFA-LoRA as
client count increases (paper Table 1 rows), and Figure-3-style convergence
should show stronger 50-client behavior. All cells use the authors'
`test_glue.yaml` hyperparameters (30 rounds, 20 local batches, bs=32,
tok_len=128, fp32, lr=0.005, weight_decay=0.0002, lora_alpha=32,
lora_dropout=0.1). Seeds are passed at runtime via `SEED=…` (see
`scripts/smoke_supplement.sh`); method selected via `SLS_ALTERNATION_MODE`.

| ID | Dataset | Model | Methods | Clients | Rank | Seeds | Config | Status |
|---|---|---|---|---:|---:|---|---|---|
| R0 | QNLI | RoBERTa-base | LoRA, FFA-LoRA, RoLoRA | 3 | 4 | 0 | `experiments/configs/table1_local_pilot.yaml` | Local pilot done |
| R1 | QNLI | RoBERTa-base | LoRA, FFA-LoRA, RoLoRA | 3 | 4 | 0 | `experiments/configs/table1_local_medium.yaml` | Medium: RoLoRA done; all modes pending |
| R2 | QNLI | RoBERTa-Large | RoLoRA | 3 | 4 | 0 | `experiments/configs/roberta_large_feasibility.yaml` | Feasibility done locally on MPS (`results/roberta_large_feasibility_rolora.log`); cluster feasibility pending real Slurm headers |
| **R3** | **QNLI** | **RoBERTa-Large** | **LoRA, FFA-LoRA, RoLoRA** | **3** | **4** | **0,1,2** | `experiments/configs/repro_qnli_c3_r4.yaml` | **Pending — paper-cell C1** |
| **R4** | **QNLI** | **RoBERTa-Large** | **LoRA, FFA-LoRA, RoLoRA** | **20** | **4** | **0,1,2** | `experiments/configs/repro_qnli_c20_r4.yaml` | **Pending — paper-cell C2** |
| **R5** | **QNLI** | **RoBERTa-Large** | **LoRA, FFA-LoRA, RoLoRA** | **50** | **4** | **0,1,2** | `experiments/configs/repro_qnli_c50_r4.yaml` | **Pending — paper-cell C3** |
| **R6** | **QNLI** | **RoBERTa-Large** | **LoRA, FFA-LoRA, RoLoRA** | **50** | **8** | **0,1,2** | `experiments/configs/repro_qnli_c50_r8.yaml` | **Pending — paper-cell C4 (rank-8 50-client row)** |

Total R3-R6 = 36 jobs (4 cells × 3 methods × 3 seeds) on QNLI. R5 and R6 are
the load-bearing client-scaling rows; if cluster time runs short, complete
those two first and treat R3/R4 as smaller-budget confirmations.

## Stretch reproduction

After QNLI (R3-R6) is fully ledgered, expand. MNLI is the canonical second
dataset; the remaining three Table-1 tasks come only after that.

| ID | Dataset | Model | Methods | Clients | Rank | Seeds | Status | Evidence |
|---|---|---|---|---:|---:|---|---|---|
| M3 | MNLI | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 3 | 4 | 0,1,2 | Pending; needs `mnli2json.py` | TBD |
| M4 | MNLI | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 20 | 4 | 0,1,2 | Pending | TBD |
| M5 | MNLI | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 50 | 4 | 0,1,2 | Pending | TBD |
| M6 | MNLI | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 50 | 8 | 0,1,2 | Pending | TBD |
| S1 | SST-2 | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 50 | 4 | 0 | Pending; needs `sst22json.py` | TBD |
| S2 | QQP | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 50 | 4 | 0 | Pending; needs `qqp2json.py` | TBD |
| S3 | RTE | RoBERTa-Large | LoRA, FFA-LoRA, RoLoRA | 50 | 4 | 0 | Pending; needs `rte2json.py` | TBD |

Stretch runs must not displace the QNLI minimum credible reproduction.

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
