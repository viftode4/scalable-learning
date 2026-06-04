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
   - Extreme heterogeneity toy results from Daniel's branch, replotted in the
     evidence tree.
   - Clear warning that local RoBERTa-base/QNLI is pipeline evidence only.

6. **Paper-scale reproduction**
   - RoBERTa-Large feasibility.
   - Selected Table 1 cells or cleanly ledgered blockers.
   - Figure-3-style 50-client convergence if compute permits; otherwise use
     RoBERTa-base proxy curves under an explicit limitation.

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
| Figure 3: toy heterogeneity baselines | C7 | `uv run python scripts/plot_toy_heterogeneity.py`; `evidence/toy_heterogeneity_20260603/figures/toy_g_heterogeneous_baselines_curves.png` | supported-toy |
| Figure 4: orthogonal-A toy paired delta | C7 | `uv run python scripts/plot_toy_heterogeneity.py`; `evidence/toy_heterogeneity_20260603/figures/toy_h_orthogonal_a_paired_delta.png` | supported-toy |
| Figure 5: RoBERTa-base 50-client proxy convergence | C6 | `uv run python scripts/plot_wandb_proxy.py`; `evidence/wandb_qnli_c50_r4_20260603/figures/proxy_a_server_accuracy_convergence.png` | supported-proxy |
| Figure 6: RoLoRA proxy LR sweep / seed variability | C6 | `proxy_c_rolora_lr_sweep.png`, `proxy_d_rolora_per_seed_trajectories.png` | supported-proxy |
| Table 1: local harness summary | C0/C2 | `make table1-pilot-summary`, `make table1-medium-summary` | partial |
| Table 2: W&B proxy baseline summary | C6 | `evidence/wandb_qnli_c50_r4_20260603/figures/proxy_plot_summary.md` | supported-proxy |
| Table 3: toy heterogeneity + orthogonal-A summary | C7 | `evidence/toy_heterogeneity_20260603/figures/toy_plot_summary.md` | supported-toy |
| Table 4: RoBERTa-Large feasibility / paper-scale attempt | C3/C4 | `make roberta-large-feasibility MODE=rolora` or one ledgered cluster attempt | planned/partial |
| Figure 7: phase-dynamics diagnostics | C5 | `make diagnostics-summary PREFIX=<run>` | planned |
| Table 5: improvement ablations beyond toy | C5/C8/C9/C10 | `SLS_LORA_INIT=orthogonal_a`; `SLS_LORA_LR_A/B`; `experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2*.yaml`; running orthogonal-A proxy seed 0 | running/code-ready |
| Table 6: limitations and failed runs | all | `experiments/ledger/README.md` | ongoing |

## Current tracking status — 2026-06-03

| Area | Status | Next action |
|---|---|---|
| Local sanity | Supported locally | Keep `make check` green before any new code/run. |
| Toy heterogeneity | Supported-toy with 5-seed CI plots | Use as mechanism/improvement motivation, not as GLUE evidence. |
| RoBERTa-base QNLI/C50 proxy | Supported-proxy with W&B plots | Use as the GLUE-scale control for orthogonal-A transfer. |
| RoBERTa-Large reproduction | Partial/planned | Make one clean attempt or ledger blocker; do not block the June-11 draft on full Table 1. |
| Improvements | Toy orthogonal-A supported; orthogonal-A proxy seed 0 running; A/B LR and FedOpt command-ready | Monitor `results/overnight_proxy_orth_a_c50_r4_lr1e-2_seed0.log`; run seeds 1/2 if competitive, otherwise A/B LR seed 0. |
| Writing | Ready for June-11 compute-constrained draft | Fill prose from `report/draft_results_brief_20260603.md` while proxy improvement run finishes. |

## June-11 draft positioning

The draft should not claim full paper reproduction. The honest draft thesis is:

> We perform a compute-constrained reproduction and proxy-scale improvement
> study of RoLoRA. We audit reproducibility issues in the supplement, validate
> the patched harness locally, show RoLoRA's robustness under extreme toy
> heterogeneity and RoBERTa-base 50-client QNLI proxy runs, and motivate
> orthogonal-A initialization as the first proposal-compatible improvement.

The current W&B figures are useful draft evidence only under this caption
constraint: **QNLI / RoBERTa-base / 50 clients / rank 4 / 20 rounds; not
RoBERTa-Large Table-1 reproduction.** The improvement section should use the
toy orthogonal-A result as supported evidence now, and add the running
orthogonal-A proxy result only after `results/overnight_proxy_orth_a_c50_r4_lr1e-2_seed0.log` contains final/server metrics.

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
