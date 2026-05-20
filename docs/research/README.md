# Source documents

Canonical references for the project. Do not edit these in place — they are snapshots.

| File | What it is |
|---|---|
| `paper-rolora.pdf` | Chen et al., *Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA*, NeurIPS 2025. The paper we reproduce. |
| `project-proposal.pdf` | Course-submitted proposal *Reproducing and Improving RoLoRA* by Popovici / Iftode / Zele, 12 May 2026. Commits to three improvement directions (orthogonal/SVD init, separate LRs for A/B, adaptive server-side optimizer) and to using the authors' released code. |
| `lecture-01-introduction.pdf` | CS 4725 lecture-1 slides by Atasu & Chen — assessment split, deadlines, paper-selection rules, cluster-access pointer. |
| `deep-research-plan.md` | Independent technical-decision document covering code availability, GPU-hour budgets, week-by-week roadmap, and risks. Treated as authoritative for compute planning. Note: its recommended improvement angle (partial participation + comm-time-aware scheduling) is **not** the angle the team submitted in the proposal; see `project-proposal.pdf` for what we actually committed to. |
| `literature-snapshot-2026-05-20.md` | External literature positioning for the 12/10 / paper-track improvement story; explains why the project should frame the proposal improvements as phase-specific A/B dynamics rather than broad federated-LoRA novelty. |

## Format note
Both Markdown and PDF were requested where applicable. In practice:
- The deep-research plan is already markdown; no PDF export is kept (would be redundant, hard to diff).
- The paper is a published PDF; no markdown export — reading and citing the PDF is the standard workflow.
- The proposal is a LaTeX-rendered PDF; a markdown export can be added later if the team shares the `.tex` source.
