# Scalable Learning Systems — RoLoRA Reproduction & Extension

TU Delft CS 4725 research seminar, Spring 2026. 10 weeks. Compute: DelftBlue + DAIC + local.

## Team
- Vlad Iftode
- Daniel Popovici
- Sorin Zele

## Paper
Chen, Guo, Ju, Dalal, Zhu, Khisti. *Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA.* NeurIPS 2025.
Local copy: [`docs/research/paper-rolora.pdf`](docs/research/paper-rolora.pdf) · OpenReview: `u4mobiHTJl`.

## Source documents
- [`docs/research/paper-rolora.pdf`](docs/research/paper-rolora.pdf) — the paper itself.
- [`docs/research/project-proposal.pdf`](docs/research/project-proposal.pdf) — our course-submitted proposal (12 May 2026).
- [`docs/research/deep-research-plan.md`](docs/research/deep-research-plan.md) — independent technical-decision document: code availability, compute budget, ranked improvement angles, week-by-week roadmap.

## Layout
```
docs/        Source documents, kickoff agenda, decision log
code/        Harness fork + our additions (RoLoRA, FFA-LoRA, baselines)
experiments/ YAML configs that map to runs
notebooks/   MNIST toy (Fig. 2 sanity check) and exploration
slurm/       DelftBlue + DAIC job scripts
results/     Output artifacts (gitignored)
report/      LaTeX writeup
```

## Status
Pre-kickoff — repo is scaffold only. See [`docs/kickoff.md`](docs/kickoff.md) for open questions to settle in the first meeting.

## Quickstart (post-kickoff, once code lands)
TBD — will be populated when the harness fork is wired up. See [`code/README.md`](code/README.md).
