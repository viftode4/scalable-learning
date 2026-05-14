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
scripts/     Setup and run utilities (dataset prep, supplement extraction/smoke)
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

To enable the authors' OpenReview harness after downloading the zip:
```bash
make supplement
make install-supplement
make supplement-smoke-all
```

See [`docs/setup/environment.md`](docs/setup/environment.md) for the full setup, [`docs/setup/openreview-supplement.md`](docs/setup/openreview-supplement.md) for fetching the authors' code, [`docs/setup/delftblue.md`](docs/setup/delftblue.md) for cluster access (TA-driven).

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
| `make local-smoke` | Full fast local evidence chain: checks, MNIST smoke, supplement smoke-all. |
| `make full-local` | Strongest laptop-feasible evidence chain: checks, 200-round MNIST, supplement smoke-all. |
| `make clean` | Remove local outputs/caches while preserving tracked placeholders. |

## Status
**Week 3 — pre-launch.** Main env is pinned, MNIST sanity checks run locally, and the authors' supplement is installed in an isolated Python 3.9 env with a tiny RoBERTa-base smoke config. Full RoBERTa-Large reproduction starts once DelftBlue/DAIC access is available. See [`docs/kickoff.md`](docs/kickoff.md) for the remaining team/process items.
