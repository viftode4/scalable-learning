# Agent / Assistant Guidance — `scalable-learning`

## What this repo is
TU Delft CS 4725 research seminar (10 weeks, 3 people) to reproduce and extend RoLoRA (Chen et al., NeurIPS 2025). See `README.md` for orientation and `docs/research/` for canonical source docs.

## Hard constraints
- **No TrustChain framing** — the user explicitly excluded this angle from this project.
- Compute is **DelftBlue + DAIC + local laptops**. Plan from low-cost to high-cost; do not propose Llama-2-7B work as a baseline path — it's a stretch goal only.
- Team is 3 people. Effort should be split **by ownership layer** (infrastructure / algorithm / improvement & analysis), not by experiment.

## Where context lives
- `docs/research/paper-rolora.pdf` — ground truth for paper claims.
- `docs/research/project-proposal.pdf` — what we promised the course.
- `docs/research/deep-research-plan.md` — authoritative for compute budgets, code-availability assessment, and ranked improvement angles.
- `docs/kickoff.md` — open questions for the team meeting.
- `docs/decisions/` — ADR-style log for decisions worth preserving (commit submodule choice, improvement angle ratification, etc.).

## Code conventions (to follow once code lands)
- Python 3.10+, PyTorch 2.x.
- Config-driven experiments: YAML under `experiments/configs/`, never flag-pollute training scripts.
- Assert exact-aggregation invariants during dev: `torch.equal(client_i.A, server.A)` at the top of every odd round (and the analogous B-check on even rounds). Remove for speed after the harness is locked in.
- Harness: fork `Pengxin-Guo/FedSA-LoRA`; do **not** invest in FederatedScope-LLM as the main vehicle.
- Watch out: there is an unrelated EMNLP'24 paper also named *RoLoRA* (quantization, `HuangOwen/RoLoRA`). Different work, do not clone.

## Improvement angle (primary, per deep-research)
Robust RoLoRA under **partial client participation** with **communication-time-aware scheduling**. The paper explicitly defers partial participation in Appendix A2. Hypothesis: stale projection matrices (A) harm performance more than stale linear heads (B).
