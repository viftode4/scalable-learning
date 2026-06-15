# Decision log (ADRs)

Append a short markdown file here for any decision worth preserving:
- Harness fork strategy (submodule vs. vendor)
- Improvement-angle ratification after kickoff
- Compute partition choices
- Departures from the deep-research plan

Format: `NNNN-short-title.md`, lead with **Context / Decision / Consequences**.

## Current key decisions

- `0005-unified-phase-dynamics-thesis.md` — frames the three proposal improvements as one phase-specific A/B dynamics thesis for the final project.
- `0006-supplement-reproducibility-gap.md` — the supplement ships SGD lr=0.005 (chance on QNLI); AdamW lr=5e-4 recovers it. Use AdamW + corrected round counts before any cluster run.
- `0007-partial-participation-ruled-out.md` — partial participation is closed by RoLoRA's exactness argument (Eqs 3-4 survive sampling); not an improvement angle.

For the synthesised "what we tried / verdict / evidence" view, see [`../results-scoreboard.md`](../results-scoreboard.md).
