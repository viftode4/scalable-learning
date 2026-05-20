# ADR 0005 — Unified phase-specific dynamics thesis

**Status:** Accepted (2026-05-20)

## Context

The submitted proposal commits to three RoLoRA-preserving improvement directions:

1. improved initialization for the down-projection matrix `A`,
2. separate learning rates for `A` and `B`,
3. adaptive server-side optimization instead of plain averaging.

Individually, these can look like disconnected parameter tweaks. The project is aiming for a top-grade final report and possibly a professor-interesting workshop-paper direction, so the improvements need one coherent research story.

A literature snapshot on 2026-05-20 showed that broad federated-LoRA aggregation novelty is crowded by recent work such as LoRA+, ADF-LoRA/TAD-LoRA, RD-LoRA, FedRot-LoRA, LA-LoRA, SDFLoRA, and FedMomentum. This makes it risky to claim broad novelty for “better federated LoRA aggregation.”

## Decision

Frame the project’s improvement phase around one thesis:

> RoLoRA removes factor-averaging aggregation error through exact alternating aggregation, but its performance is governed by phase-specific behavior of the `A` and `B` factors.

Under this thesis:

- improved initialization tests whether the starting `A` subspace controls early convergence and variance;
- separate learning rates test whether RoLoRA’s alternating `A` and `B` phases need different optimization scales;
- adaptive server-side optimization tests whether active-factor updates should be accumulated differently across rounds.

The project will prioritize a credible RoBERTa-Large reproduction first, then run the smallest improvement grid that can answer this phase-specific question.

## Consequences

- The final report should not present three unrelated improvements. It should present a single phase-dynamics investigation with three targeted interventions.
- Negative results remain useful if they diagnose phase behavior, convergence speed, variance, or update instability.
- The team should avoid broad novelty claims in heterogeneous-rank, DP, decentralized, or generic LoRA aggregation spaces unless deliberately positioning against the newer literature.
- Experiment logging must include phase markers and per-round metrics so the team can explain A-phase versus B-phase behavior.

## Alternatives rejected

- **Treat each proposal improvement independently.** Rejected: likely to read like an unfocused sweep rather than a research contribution.
- **Pivot fully to partial participation.** Rejected for now: it is not the submitted proposal direction and recent decentralized/heterogeneous alternating-LoRA work makes broad novelty claims harder.
- **Try to reproduce all Table 1 cells before any analysis.** Rejected: full breadth may consume all compute without producing an original insight. One dataset done deeply is a safer path to a high-grade report.
