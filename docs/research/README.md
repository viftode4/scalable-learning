# Source documents

Canonical references for the project. Do not edit these in place — they are snapshots.

| File | What it is |
|---|---|
| `paper-rolora.pdf` | Chen et al., *Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA*, NeurIPS 2025. The paper we reproduce. |
| `project-proposal.pdf` | Course-submitted proposal *Reproducing and Improving RoLoRA* by Popovici / Iftode / Zele, 12 May 2026. Commits to three improvement directions (orthogonal/SVD init, separate LRs for A/B, adaptive server-side optimizer) and to using the authors' released code. |
| `lecture-01-introduction.pdf` | CS 4725 lecture-1 slides by Atasu & Chen — assessment split, deadlines, paper-selection rules, cluster-access pointer. |
| `deep-research-plan.md` | Independent technical-decision document covering code availability, GPU-hour budgets, week-by-week roadmap, and risks. Treated as authoritative for compute planning. Note: its recommended improvement angle (partial participation + comm-time-aware scheduling) is **not** the angle the team submitted in the proposal; see `project-proposal.pdf` for what we actually committed to. |
| `literature-snapshot-2026-05-20.md` | External literature positioning for the 12/10 / paper-track improvement story; explains why the project should frame the proposal improvements as phase-specific A/B dynamics rather than broad federated-LoRA novelty. |
| `paper-frlora-iclr2025.pdf` | Yan et al., *Federated Residual Low-Rank Adaptation of Large Language Models*, ICLR 2025. Added 2026-06-08 as an SVD/PiSSA-style initialization and residual-update reference for proposal improvement axis 1. |
| `paper-pissa-arxiv2404.02948.pdf` | Meng et al., *PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models*, arXiv/NeurIPS 2024 version. Added 2026-06-08 as the main compensated SVD LoRA-initialization reference. |
| `paper-lora-a2-arxiv2410.22815.pdf` | Koo et al., *Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients* (LoRA-A2). Cited by RoLoRA as concurrent alternating-optimization work. Added 2026-06-11 for the basis-transport novelty check: it alternates freeze + adaptive rank selection, but performs no cross-factor correction when the trained factor changes; it explicitly dismisses product-space reconciliation as "computationally unstable" (§4, Eq. 3-6). |
| `paper-adf-lora-arxiv2511.18291.pdf` | Wang et al., *ADF-LoRA: Alternating Low-Rank Aggregation for Decentralized Federated Fine-Tuning*. Added 2026-06-11 for the basis-transport novelty check: its fixes are interval-based phase switching and joint mixing of both blocks to keep the frozen factor aligned under peer-to-peer gossip — a DFL-only alignment problem; no coefficient re-expression after a basis update. |

## Format note
Both Markdown and PDF were requested where applicable. In practice:
- The deep-research plan is already markdown; no PDF export is kept (would be redundant, hard to diff).
- The paper is a published PDF; no markdown export — reading and citing the PDF is the standard workflow.
- The proposal is a LaTeX-rendered PDF; a markdown export can be added later if the team shares the `.tex` source.
