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
scripts/     One-off prep utilities (dataset prep, supplement extraction)
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
uv sync
uv run python notebooks/mnist_fig2.py
```

See [`docs/setup/environment.md`](docs/setup/environment.md) for the full setup, [`docs/setup/openreview-supplement.md`](docs/setup/openreview-supplement.md) for fetching the authors' code, [`docs/setup/delftblue.md`](docs/setup/delftblue.md) for cluster access (TA-driven).

## Status
**Week 3 — pre-launch.** Repo scaffolded, env pinned, MNIST sanity check runnable on a laptop. RoBERTa training entrypoint and the improvement experiments start in W4. See [`docs/kickoff.md`](docs/kickoff.md) for the open items.
