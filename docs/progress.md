# Project progress board

This is the live control surface for the RoLoRA final project. Keep it short,
current, and evidence-backed. Every row needs a concrete next action and an
evidence path.

## Current thesis

We reproduce RoLoRA's core client-scaling claim, then characterize and improve
phase-specific `A`/`B` dynamics using minimal changes that preserve exact
alternating aggregation.

## Strategy lock

- **Primary path:** reproduction-first, diagnostics-backed.
- **Fallback path:** diagnostic-first smaller-scale story if paper-scale compute
  is blocked.
- **Primary dataset:** MNLI.
- **Fallback dataset:** QNLI, because the supplement pipeline is already wired
  and locally verified there.
- **Forbidden pivots:** unrelated prior-project framing, a different paper-presentation paper,
  TOPdesk-before-TA cluster requests, or Llama-2-7B as a baseline path.

## Fallback triggers

Switch from paper-scale reproduction-first to smaller diagnostic-first execution
when any trigger below fires:

| Trigger | Default response |
|---|---|
| No usable DelftBlue/DAIC path by mid-W4 | Continue RoBERTa-base/QNLI diagnostics and prepare cluster-ready configs. |
| RoBERTa-Large feasibility fails due to memory/runtime | Record failure in the ledger; run QNLI/RoBERTa-base or smaller RoBERTa-Large cells with full diagnostics. |
| 3-client RoBERTa-Large baseline misses the paper by more than ±2% without an explainable setup cause | Stop broad scaling; debug baseline comparability before more cluster spend. |
| 50-client degradation pattern is not reproduced by the agreed cutoff | Prioritize phase-dynamics diagnostics and transparent negative-result analysis. |

## Change log

| Date | What changed | Why it matters | Evidence |
|---|---|---|---|
| 2026-05-20 | Added fallback triggers, lane-based workstreams, and claim ledger. | Makes the 12/10 plan visible and reviewable. | `docs/progress.md` |
| 2026-05-20 | Added dataset rule, compute gates, and improvement comparability constraints. | Prevents vague “run more experiments” drift. | `docs/experiment-matrix.md` |
| 2026-05-20 | Added report skeleton and figure/table placeholders. | Forces each experiment to fill a report slot. | `report/README.md` |
| 2026-05-20 | Added RoBERTa-Large feasibility config and Make targets. | Creates a safe gate before cluster-scale reproduction. | `experiments/configs/roberta_large_feasibility.yaml`, `Makefile` |
| 2026-05-20 | Added diagnostics summary mode. | Starts the phase-dynamics evidence path before richer supplement instrumentation. | `scripts/summarize_supplement.py` |

## Workstreams

Owners stay as lanes until the team maps names to work. This keeps the setup
usable now without pretending ownership is decided.

| Workstream | Lane owner | Current status | Next action | Evidence path | Blocker / risk |
|---|---|---|---|---|---|
| Infrastructure & baselines | Setup lane | Local env, supplement install path, tests, smoke runs, and pilot summaries exist. | Run `make table1-medium-all`; summarize with `make table1-medium-summary`. | `experiments/ledger/README.md`, `results/table1_*.log` | RoBERTa-Large cluster access is still TA-driven. |
| Harness escalation | Setup lane | Local QNLI RoBERTa-base configs exist; paper-scale config is now split into a feasibility gate. | Run `make roberta-large-feasibility MODE=rolora` only on a GPU-capable machine. | `experiments/configs/roberta_large_feasibility.yaml`, `results/roberta_large_feasibility_*.log` | MNLI/data prep and GPU availability. |
| Algorithm & ablations | Algorithm lane | RoLoRA/LoRA/FFA-LoRA modes run through the patched supplement; invariant tests exist for local helper code. | Add/verify active-factor update norms and frozen-factor markers before serious runs. | `code/harness/rolora-supplement.patch`, `tests/` | Supplement code is gitignored; patch discipline matters. |
| Improvement & analysis | Analysis lane | Three proposal directions are selected; unified phase-specific thesis recorded in ADR 0005. | Run only the smallest phase-dynamics grid after diagnostics are present. | `docs/research/literature-snapshot-2026-05-20.md`, `docs/experiment-matrix.md` | A/B LR novelty is weak unless paper ablations are acknowledged. |
| Report & presentation | Analysis lane | Paper presentation outline exists; report now has a claim-led skeleton. | Fill the claim ledger as each run completes or fails. | `report/README.md`, `docs/templates/paper-presentation-outline.md` | Writing too late will make results look like an experiment dump. |
| Cluster / access | Setup lane | TA-driven access instructions not yet received; provisional docs exist. | Monitor BrightSpace/course email; update setup docs once TA instructions arrive. | `docs/setup/delftblue.md`, `slurm/README.md` | Do not file TOPdesk blindly; partition info is unconfirmed. |

## Claim ledger

Every report claim must map to evidence. Keep unsupported claims in `planned`,
not in prose.

| Claim ID | Claim | Status | Required evidence | Config / command | Seeds | Log / plot | Owner | Reviewer | Limitations |
|---|---|---|---|---|---|---|---|---|---|
| C0 | The local harness preserves RoLoRA/LoRA/FFA-LoRA execution modes. | supported-local | Smoke logs with `[sls-rolora]` markers. | `make supplement-smoke-all` | 0 | `results/smoke_*.log` | Setup lane | TBD | Pipeline evidence only. |
| C1 | The local toy reproduces the qualitative Figure-2 ordering. | supported-local | MNIST plot and final accuracies. | `make mnist-paper` | 0 | `results/mnist_fig2.png` | Analysis lane | TBD | Toy model, not GLUE. |
| C2 | RoLoRA is comparable to or better than LoRA/FFA-LoRA at the local Table-1-shaped rung. | running | Medium all-mode summary. | `make table1-medium-all && make table1-medium-summary` | 0 | `results/table1_medium_*.log` | Setup lane | TBD | RoBERTa-base/QNLI only. |
| C3 | RoBERTa-Large MNLI feasibility is known before cluster spend. | planned | One tiny feasibility run or actionable failure. | `make roberta-large-feasibility MODE=rolora` | 0 | `results/roberta_large_feasibility_rolora.log` | Setup lane | TBD | Requires GPU/data readiness. |
| C4 | RoLoRA degrades less than LoRA/FFA-LoRA as clients increase. | planned | 3/20/50-client RoBERTa-Large table and curve. | R3-R5 matrix rows | 0, then 0/1/2 for 50 clients | TBD | Algorithm lane | TBD | Full Table 1 may be compute-blocked. |
| C5 | Phase-specific diagnostics explain at least one improvement or null result. | planned | Per-round phase, update norm, metric, and wall-time traces. | I1-I5 matrix rows | 0 first, replicate winner | TBD | Analysis lane | TBD | Requires supplement instrumentation beyond current final metrics. |

## This-week checklist

- [x] Make the 12/10 plan visible in README/progress/matrix/report docs.
- [x] Add the RoBERTa-Large feasibility config and command target.
- [x] Add diagnostics summary parsing for existing supplement logs.
- [ ] Run `make table1-medium-all` or explicitly defer with reason.
- [ ] Summarize local medium logs with `make table1-medium-summary`.
- [ ] Run `make diagnostics-summary PREFIX=table1_medium` after logs exist.
- [ ] Try `make roberta-large-feasibility MODE=rolora` on the first GPU-capable machine.
- [ ] Record every experiment attempt, including failures, in `experiments/ledger/README.md`.
- [ ] Map human names to setup / algorithm / analysis lanes.

## Evidence rules

- Every result claim needs a command, config, seed, log/plot path, and interpretation.
- Local RoBERTa-base pilot metrics are pipeline evidence only; do not compare them directly to paper Table 1.
- Failed runs are evidence. Keep the log and record the blocker.
- No improvement claim is report-ready unless it has a baseline, a curve, and a phase-dynamics explanation.
