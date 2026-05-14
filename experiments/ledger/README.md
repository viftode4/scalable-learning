# Experiment ledger

This ledger records local and cluster evidence as we run it. Keep entries short,
copy-pastable, and honest about scale. Generated logs stay in `results/` and are
not committed.

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
3. Add a RoBERTa-large 1-round feasibility probe if the medium run is stable.
4. Move the first paper-comparable RoBERTa-Large 3-client cell to DelftBlue/DAIC when access is ready.

## Rules

- Do not compare `table1_local_*` metrics directly to paper Table 1.
- Every committed claim should cite a command and a log/plot path.
- Before cluster runs, keep `make check` green.
