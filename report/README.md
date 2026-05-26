# Final report skeleton

This directory holds the final CS4725 project writeup. Build artifacts
(`build/`, `*.aux`, etc.) are gitignored. The report should be filled as the
claim ledger in `docs/progress.md` moves from planned to supported/failed.

## One-sentence thesis

RoLoRA's exact alternating aggregation gives the core robustness benefit, but
its final behavior is governed by phase-specific `A`/`B` dynamics that can be
characterized and possibly improved with initialization, A/B learning-rate, and
active-factor server-optimization choices.

## Required report sections

1. **Introduction**
   - Problem: federated LoRA averaging is biased because averaging factors is
     not averaging products.
   - Contribution: reproduce RoLoRA and study phase-specific A/B dynamics under
     limited compute.

2. **Background**
   - LoRA factorization.
   - Federated LoRA aggregation trap.
   - RoLoRA alternating optimization.
   - Why A/B phases are a meaningful intervention surface.

3. **Reproduction protocol**
   - Source paper and OpenReview supplement.
   - Datasets, clients, ranks, seeds.
   - Deviations from paper caused by compute.
   - Exact command/config/log discipline.

4. **Reproducibility audit of the OpenReview supplement** *(see ADR 0006)*
   - The federated RoLoRA paper has no public github repository; the
     OpenReview supplement is the sole released artifact.
   - The shipped `test_glue.yaml` uses SGD `lr=0.005` (Adam line is
     commented out). At that recipe, neither our local RoBERTa-base
     QNLI runs nor Daniel's cluster RoBERTa-Large runs learn beyond
     chance in 30 rounds. Replacing with AdamW `lr=5e-4` reaches
     test_acc ≥ 0.87 at 40 rounds on RoBERTa-base QNLI.
   - The supplement's trainer permanently freezes the SEQ_CLS head
     from `step_count==0` onward, with no documentation. Empirically
     the freeze is harmful but not catastrophic (control run with
     AdamW + frozen head still reaches ≥ 0.82 by round 9, because
     LoRA adapts upstream features into the random head's effective
     decision direction).
   - Report frames the two findings separately: (a) **shipped
     optimiser cannot reproduce paper accuracies** — strong empirical
     claim; (b) **undocumented classifier-freeze** — code-quality
     concern that slows but doesn't block learning.
   - Patches are recorded in `code/harness/rolora-supplement/`
     `federatedscope/llm/trainer/trainer.py` and `client.py` on
     branch `fix-rolora` (commits `8c60faa`, `3e5f68e`); the
     empirical-evidence table sits in ADR 0006 and the change-log row
     for 2026-05-27 in `docs/progress.md`.

5. **Local sanity evidence**
   - MNIST Figure-2-style sanity.
   - Supplement smoke and Table-1-shaped local pilot.
   - Clear warning that local RoBERTa-base/QNLI is pipeline evidence only.

6. **Paper-scale reproduction**
   - RoBERTa-Large feasibility.
   - Selected Table 1 cells.
   - Figure-3-style 50-client convergence if compute permits.

7. **Phase diagnostics**
   - A/B phase markers.
   - Per-round metrics.
   - Update norms and frozen-factor markers.
   - Wall-clock and failure evidence.

8. **Improvement experiments**
   - Orthogonal/data-informed A initialization.
   - Separate A/B learning rates, acknowledging the paper's 2×/4× LR ablations.
   - Active-factor server momentum/Adam.
   - Combined best only if individual axes show signal.

9. **Discussion and limitations**
   - What reproduced, what did not, and why.
   - What diagnostics explain.
   - Compute limits and external-validity limits.
   - Why no unrelated prior-project framing or partial-participation pivot in the main story.

10. **Conclusion**
   - Reproduction status.
   - Strongest insight.
   - Future work.

## Figure and table placeholders

| Artifact | Claim ID | Source command / config | Status |
|---|---|---|---|
| Figure 1: RoLoRA alternating mechanism diagram | C0 | drawn from paper explanation | planned |
| Figure 2: MNIST sanity plot | C1 | `make mnist-paper` | supported-local |
| Table 1: local harness summary | C0/C2 | `make table1-pilot-summary`, `make table1-medium-summary` | partial |
| Table 2: RoBERTa-Large feasibility result | C3 | `make roberta-large-feasibility MODE=rolora` | planned |
| Table 3: selected Table-1 reproduction cells | C4 | R3-R5 matrix rows | planned |
| Figure 3: 50-client convergence curves | C4 | R5/R6 logs | planned |
| Figure 4: phase-dynamics diagnostics | C5 | `make diagnostics-summary PREFIX=<run>` | planned |
| Table 4: improvement ablations | C5 | I1-I5 matrix rows | planned |
| Table 5: limitations and failed runs | all | `experiments/ledger/README.md` | ongoing |

## Current tracking status — 2026-05-20

| Area | Status | Next action |
|---|---|---|
| Local sanity | Supported locally | Keep `make check` green before large runs. |
| Local Table-1-shaped rung | Partial | Run `make table1-medium-all`; update C2. |
| Diagnostics | Parser scaffold ready | Instrument update norms, frozen-factor equality, wall time, and memory. |
| RoBERTa-Large feasibility | Config ready, run pending | Run `make roberta-large-feasibility MODE=rolora` on GPU. |
| Improvements | Planned | Start only after diagnostics can explain positive/negative outcomes. |
| Writing | Skeleton ready | Fill sections as claim-led evidence arrives. |

## Done criteria before W8 draft

- Every section has at least bullet content.
- Every planned claim appears in `docs/progress.md` claim ledger.
- Every figure/table placeholder has a source command or an explicit blocker.
- Unsupported claims are labeled planned or failed, not written as conclusions.

## Done criteria before W9 final

- Every final claim has command, config, seed, log/plot, and interpretation.
- Negative results are included when they explain phase dynamics.
- The paper's own asymmetric-LR ablations are acknowledged.
- The reproducibility appendix lists the supplement patch, configs, Make targets,
  and exact rerun commands.
