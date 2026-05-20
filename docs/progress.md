# Project progress board

This board is the live control surface for the RoLoRA final project. Keep it short, current, and evidence-backed. Every row should have a concrete next action and an evidence path.

## Current thesis

We reproduce RoLoRA’s core client-scaling claim, then characterize and improve phase-specific `A`/`B` dynamics using minimal changes that preserve exact alternating aggregation.

## Workstreams

| Workstream | Owner | Current status | Next action | Evidence path | Blocker / risk |
|---|---|---|---|---|---|
| Infrastructure & baselines | TBD | Local env, supplement install path, tests, smoke runs, and pilot summaries exist. | Run `make table1-medium-all`; summarize with `make table1-medium-summary`. | `experiments/ledger/README.md`, `results/table1_*.log` | RoBERTa-Large cluster access still TA-driven. |
| Algorithm & ablations | TBD | RoLoRA/LoRA/FFA-LoRA modes run through the patched supplement; invariant tests exist for local helper code. | Add/verify logging for phase, active factor, update norms, and per-round metrics before large runs. | `code/harness/rolora-supplement.patch`, `tests/` | Supplement code is gitignored; patch discipline matters. |
| Improvement & analysis | TBD | Three proposal directions are selected; unified phase-specific thesis recorded in ADR 0005. | Specify the smallest improvement grid for init, A/B LR, and server optimizer. | `docs/research/literature-snapshot-2026-05-20.md`, `docs/experiment-matrix.md` | Risk of disconnected sweeps if thesis is not enforced. |
| Report & presentation | TBD | Paper presentation outline exists; final report directory is mostly empty. | Create report skeleton with figure/table placeholders and assign owners. | `docs/templates/paper-presentation-outline.md`, `report/README.md` | Writing too late will make results look like an experiment dump. |
| Cluster / access | TBD | TA-driven access instructions not yet received; provisional docs exist. | Monitor BrightSpace/course email; update setup docs once TA instructions arrive. | `docs/setup/delftblue.md`, `slurm/README.md` | Do not file TOPdesk blindly; partition info is unconfirmed. |

## This-week checklist

- [ ] Assign owners for all five workstreams.
- [ ] Confirm meeting cadence and communication channel.
- [ ] Run `make table1-medium-all` or explicitly defer with reason.
- [ ] Add RoBERTa-Large one-round feasibility config.
- [ ] Start `report/` skeleton.
- [ ] Record every experiment attempt, including failures, in `experiments/ledger/README.md`.

## Evidence rules

- Every result claim needs a command, config, seed, log/plot path, and interpretation.
- Local RoBERTa-base pilot metrics are pipeline evidence only; do not compare them directly to paper Table 1.
- Failed runs are evidence. Keep the log and record the blocker.
