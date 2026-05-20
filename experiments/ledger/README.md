# Experiment ledger

This ledger records local and cluster evidence as we run it. Keep entries short,
copy-pastable, and honest about scale. Generated logs stay in `results/` and are
not committed.

## Tracking milestones

| Date | Artifact | Evidence | Why it matters |
|---|---|---|---|
| 2026-05-20 | Paper-track project controls | `README.md`, `docs/progress.md`, `docs/experiment-matrix.md`, `docs/plans/12-10-paper-track-rolora.md`, `report/README.md` | Makes the strategy, claim ledger, compute gates, report skeleton, and remaining work visible to humans and agents. |
| 2026-05-20 | RoBERTa-Large feasibility gate | `experiments/configs/roberta_large_feasibility.yaml`, `make roberta-large-feasibility MODE=rolora` | Creates a safe GPU gate before spending cluster time on paper-scale reproduction. |
| 2026-05-20 | Diagnostics summary scaffold | `scripts/summarize_supplement.py --diagnostics`, `make diagnostics-summary PREFIX=<run>` | Starts the phase-dynamics evidence path from existing logs; update norms/frozen-factor markers still need supplement instrumentation. |

## Local evidence collected

| Date | Command | Scale | Evidence | Interpretation |
|---|---|---|---|---|
| 2026-05-14 | `make mnist-paper` | MNIST, 5 clients, rank 1, 200 rounds | `results/mnist_fig2.png`; final acc: RoLoRA `0.4794`, LoRA `0.4631`, FFA-LoRA `0.3767` | Reproduces the paper's qualitative Fig. 2 ordering locally. |
| 2026-05-14 | `make supplement-smoke-all` | RoBERTa-base QNLI, 2 clients, 2 rounds, 2 local batches | `results/smoke_*.log`; all modes emit `[sls-rolora]` markers | Authors' supplement harness + three-mode patch execute locally. |
| 2026-05-14 | `make table1-pilot MODE={rolora,lora,ffa_lora}` | RoBERTa-base QNLI, 3 clients, 3 rounds, 3 local batches | `results/table1_pilot_*.log`; summarize with `make table1-pilot-summary` | Table-1-shaped pipeline works locally, but scale is too tiny for paper-comparable numbers. |
| 2026-05-14 | `make table1-medium MODE=rolora` | RoBERTa-base QNLI, 3 clients, 10 rounds, 5 local batches | `results/table1_medium_rolora.log`; summarize with `make table1-medium-summary`; final acc: test `0.511258`, val `0.504298` | Stronger single-mode local run completes; next step is `make table1-medium-all` if we can spare the runtime. |

## Next local runs

1. Run `make table1-medium-all` if local runtime is acceptable.
2. Summarize with `make table1-medium-summary`.
3. Run `make diagnostics-summary PREFIX=table1_medium` and inspect phase/metric rows.
4. Run `make roberta-large-feasibility MODE=rolora` on a GPU-capable machine.
5. Move the first paper-comparable RoBERTa-Large 3-client cell to DelftBlue/DAIC when access is ready.

## Rules

- Do not compare `table1_local_*` metrics directly to paper Table 1.
- Every committed claim should cite a command and a log/plot path.
- Before cluster runs, keep `make check` green.
- Every non-result setup milestone that changes the experiment path should be recorded above.
