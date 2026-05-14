# ADR 0001 — Harness strategy

**Status:** Accepted (2026-05-14)

## Context

The project proposal (submitted 12 May 2026) commits in writing to:

> "We will use the authors' released code as the starting point for both reproduction and improvement experiments."

The authors' code is not on public GitHub. It exists only as a supplementary zip attached to OpenReview submission `u4mobiHTJl`. The deep-research plan flags this as likely a research-grade dump rather than a polished release.

We therefore need (a) a clear path to ingest the supplement, and (b) a vetted fallback so a broken supplement does not block the project.

## Decision

- **Primary harness:** the OpenReview supplement, extracted into `code/harness/rolora-supplement/`. This path is gitignored (author code may not be redistributable). Extraction is user-driven via `docs/setup/openreview-supplement.md` + `scripts/extract_supplement.sh`.
- **Backup harness:** our fork `viftode4/FedSA-LoRA` (upstream `Pengxin-Guo/FedSA-LoRA`, ICLR 2025), attached at `code/harness/fedsa-lora/` as a git submodule. Already implements LoRA / FFA-LoRA / FedSA-LoRA on RoBERTa+GLUE under FedAvg with Dirichlet-α splits.

## Consequences

- Teammates run `git submodule update --init --recursive` after clone — the submodule is real, in git.
- Teammates also run the supplement-extraction step from the setup guide — that piece is **not** in git, has to be fetched per-clone.
- If the supplement fails to reproduce the LoRA baseline on 3-client MNLI within ±2% of the paper (the deep-research plan's W2 kill criterion), we switch primary to FedSA-LoRA and email the authors using `docs/templates/author-email.md`.
- When we begin modifying the submodule, we create a branch `sls-rolora/main` in our fork and point the submodule at it; `main` stays tracking upstream.

## Alternatives rejected

- **Vendor the supplement in-tree (commit the unpacked code).** Rejected: redistribution rights unclear; pollutes diffs.
- **Vendor FedSA-LoRA in-tree.** Rejected: loses upstream history, makes future rebases hard.
- **Use FederatedScope-LLM as primary.** Rejected per deep-research plan: heavy, rough edges, older than FedSA-LoRA's harness.
- **Skip the supplement entirely.** Rejected: violates the written proposal commitment.
