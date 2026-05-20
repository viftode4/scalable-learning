# Scalable Learning Systems — RoLoRA Reproduction & Extension

TU Delft **CS 4725** research seminar (Spring 2026). 9 course weeks, currently in **week 3**. Hard end: week 9 final report + presentation (the only graded deliverables).

## Team
- Vlad Iftode
- Daniel Popovici
- Sorin Zele

## Course staff
- **Dr. Kubilay Atasu** — Associate Professor, coordinator (lectures, projects, presentations, homeworks).
- **Dr. Rui Wang** — Postdoctoral researcher (projects, guest lecture).
- **Dennis Heijmans** — MSc thesis student (homeworks, presentations).

## Paper
Chen, Guo, Ju, Dalal, Zhu, Khisti. *Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA.* NeurIPS 2025.
Local copy: [`docs/research/paper-rolora.pdf`](docs/research/paper-rolora.pdf) · OpenReview: `u4mobiHTJl`.

## Assessment
| Component | Weight | When |
|---|---|---|
| Paper presentation (on RoLoRA itself) | 20% | weeks 7–8; 10–12 min present + 5–6 min Q&A; rubric 20/40/25/15 |
| Research project (reproduction + improvement) | 60% | weeks 4–9; deliverables below |
| Individual homeworks | 20% | due weeks 4 and 5 |

## Deadlines
| Week | Deliverable | Status |
|---|---|---|
| W4 | Project proposal (mandatory, ungraded) | ✅ submitted 12 May 2026 |
| W6 | Midterm review meeting (mandatory, ungraded) | pending |
| W8 | Draft project report (mandatory, ungraded) | pending |
| W9 | Final project report + final presentation (mandatory, **graded**) | pending |

## Committed improvement directions (per submitted proposal)
1. **Improved initialization** — orthogonal / SVD-based init for the down-projection matrix A.
2. **Separate learning rates for A and B** — LoRA+-style asymmetric LRs, enabled by RoLoRA's per-round factor isolation.
3. **Adaptive server-side optimization** — lightweight federated optimizer in place of plain averaging.

All three preserve RoLoRA's alternating structure.

## Source documents
- [`docs/research/paper-rolora.pdf`](docs/research/paper-rolora.pdf) — the paper.
- [`docs/research/project-proposal.pdf`](docs/research/project-proposal.pdf) — our submitted proposal.
- [`docs/research/lecture-01-introduction.pdf`](docs/research/lecture-01-introduction.pdf) — CS 4725 lecture-1 slides.
- [`docs/research/deep-research-plan.md`](docs/research/deep-research-plan.md) — independent technical-decision document (compute budget, roadmap, risks).

## Layout
```
docs/        Source documents, kickoff agenda, decision log, setup guides, templates
code/        Our code + harness checkouts (FedSA-LoRA submodule, RoLoRA supplement)
experiments/ YAML configs that map to runs
notebooks/   MNIST Figure-2 sanity check and exploration
scripts/     Setup and run utilities (dataset prep, supplement extraction/smoke/summary)
slurm/       DelftBlue / DAIC job templates
results/     Output artifacts (gitignored)
report/      LaTeX writeup
tests/       pytest suite (aggregation math, invariants)
```

## Quickstart
```bash
git clone <this-repo>
cd scalable-learning
git submodule update --init --recursive
make sync
make check
make mnist-smoke
```

The authors' OpenReview supplement is vendored in this repo. To enable its isolated runtime:
```bash
make install-supplement
make supplement-smoke-all
```

`make supplement` remains available only to refresh the vendored copy from the original OpenReview zip.

See [`docs/setup/environment.md`](docs/setup/environment.md) for the full setup, [`docs/setup/openreview-supplement.md`](docs/setup/openreview-supplement.md) for fetching the authors' code, [`docs/setup/delftblue.md`](docs/setup/delftblue.md) for cluster access (TA-driven), and [`experiments/ledger/README.md`](experiments/ledger/README.md) for run evidence.


## Project control docs

Use these docs to keep the final project execution visible to humans and agents:

- [`docs/progress.md`](docs/progress.md) — live owner/status/next-action board.
- [`docs/experiment-matrix.md`](docs/experiment-matrix.md) — reproduction and improvement run matrix.
- [`docs/plans/12-10-paper-track-rolora.md`](docs/plans/12-10-paper-track-rolora.md) — 12/10 + paper-track execution plan and critique.
- [`docs/research/literature-snapshot-2026-05-20.md`](docs/research/literature-snapshot-2026-05-20.md) — external literature positioning for the improvement story.
- [`docs/decisions/0005-unified-phase-dynamics-thesis.md`](docs/decisions/0005-unified-phase-dynamics-thesis.md) — ADR pinning the unified phase-specific thesis.

## Tracking snapshot — 2026-05-20

### Done / visible now

- Paper-track strategy is locked: reproduce first, diagnose phase-specific A/B dynamics, then test proposal-compatible improvements.
- `docs/progress.md` is the live dashboard and claim ledger.
- `docs/experiment-matrix.md` defines dataset rules, compute gates, reproduction rows, improvement rows, and stop/fallback criteria.
- `report/README.md` is the report skeleton with figure/table placeholders mapped to claim IDs.
- `experiments/configs/roberta_large_feasibility.yaml` defines the GPU-only RoBERTa-Large feasibility gate.
- `scripts/summarize_supplement.py --diagnostics` and `make diagnostics-summary` parse manifest, per-result metrics, and phase markers from logs.

### Left / next evidence gates

1. Run `make table1-medium-all`, then `make table1-medium-summary` and `make diagnostics-summary PREFIX=table1_medium`.
2. Ledger the medium all-mode result or failure in `experiments/ledger/README.md`.
3. Run `make roberta-large-feasibility MODE=rolora` on a GPU-capable machine.
4. Add stronger supplement instrumentation for update norms, frozen-factor equality markers, wall time, and memory.
5. Implement the improvement knobs in order: orthogonal A init, A/B LR split, active-factor server momentum.
6. Fill the report skeleton continuously; no claim should enter final prose without a claim-ledger evidence path.

## Local commands
| Command | Purpose |
|---|---|
| `make check` | Run first-party tests + lint. |
| `make mnist-smoke` | Fast MNIST sanity check. |
| `make mnist` | Default local MNIST Figure-2 run. |
| `make mnist-paper` | Stronger 200-round MNIST Figure-2 run used as the local paper-sanity check. |
| `make supplement-smoke-all` | Run the tiny supplement smoke config in `rolora`, `lora`, and `ffa_lora` modes. |
| `make table1-pilot MODE=rolora` | Run a 3-client QNLI RoBERTa-base local pilot for one mode. |
| `make table1-pilot-all` | Run the local Table-1-shaped pilot for all three modes. |
| `make table1-pilot-summary` | Parse `results/table1_pilot_*.log` into a metrics table. |
| `make table1-medium MODE=rolora` | Stronger local pilot: 3-client QNLI RoBERTa-base, 10 rounds, 5 local batches. |
| `make table1-medium-all` | Run the stronger local pilot for all three modes. |
| `make table1-medium-summary` | Parse `results/table1_medium_*.log` into a metrics table. |
| `make roberta-large-feasibility MODE=rolora` | Run the tiny GPU-only RoBERTa-Large feasibility gate before cluster reproduction. |
| `make roberta-large-feasibility-summary` | Parse feasibility logs into a metrics table. |
| `make diagnostics-summary PREFIX=table1_medium` | Parse manifest, per-round metrics, and phase markers from supplement logs. |
| `make cluster-dry-run` | Print the current cluster gate and feasibility config without submitting Slurm jobs. |
| `make local-smoke` | Full fast local evidence chain: checks, MNIST smoke, supplement smoke-all. |
| `make full-local` | Strongest laptop-feasible evidence chain: checks, 200-round MNIST, supplement smoke-all. |
| `make clean` | Remove local outputs/caches while preserving tracked placeholders. |

## What works locally now

- `make check` — first-party tests and lint.
- `make mnist-paper` — 200-round MNIST paper-sanity run; latest local result: RoLoRA `0.4794` > LoRA `0.4631` > FFA-LoRA `0.3767`.
- `make supplement-smoke-all` — authors' FederatedScope supplement runs locally in `rolora`, `lora`, and `ffa_lora` modes.
- `make table1-pilot-all` — Table-1-shaped local QNLI pilot: RoBERTa-base, 3 clients, 3 rounds, 3 local batches.
- `make table1-pilot-summary` — parses local pilot logs into a metrics table.
- `make table1-medium MODE=rolora` — stronger local pilot; verified for `rolora` on 2026-05-14. Run `make table1-medium-all` next to close the local all-mode rung.
- `make diagnostics-summary PREFIX=table1_medium` — parses existing medium logs into the diagnostic table shape; richer update-norm/frozen-factor fields still need supplement instrumentation.
- `make cluster-dry-run` — documents that cluster execution remains TA-gated rather than silently pretending Slurm is ready.

## What is not local yet

Full paper Table 1 is RoBERTa-Large across MNLI/QQP/QNLI, 3/20/50 clients, three methods, and multiple seeds. The plan estimates hundreds of GPU-hours, so local runs are pipeline and mechanism evidence, not paper-comparable Table 1 numbers. The explicit next gate is `make roberta-large-feasibility MODE=rolora` on a GPU-capable machine.

## Status
**Week 3 — pre-launch.** Main env is pinned, MNIST sanity checks run locally, the authors' supplement is installed in an isolated Python 3.9 env, and local Table-1-shaped pilots are runnable. Next local step: `make table1-medium-all` if runtime is acceptable; after that, run the explicit `make roberta-large-feasibility MODE=rolora` gate on a GPU-capable machine. Full RoBERTa-Large reproduction starts once DelftBlue/DAIC access is available. See [`docs/kickoff.md`](docs/kickoff.md) for the remaining team/process items.
