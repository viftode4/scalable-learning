# Improvement handoff — 2026-06-13

Read this first before launching more improvement runs. It separates actual
RoBERTa-base proxy evidence from the new toy triage bank.

## Bottom line

The best **actual-model** improvement evidence is still:

1. **Orthogonal-A + BBA phase schedule** — strongest result so far.
2. **Adaptive refresh + orthogonal-A** — stable positive result across 3 seeds.
3. **SVD-compensated init** — already ran on the actual QNLI proxy and is only a
   small/neutral gain over vanilla RoLoRA at the tested LR.

The new toy bank is useful for fast debugging, but it should not override the
actual RoBERTa-base proxy results.

## Actual QNLI / RoBERTa-base proxy results

Cell unless noted: QNLI, RoBERTa-base, 50 clients, rank 4, 20 rounds.

| Variant | Seeds | Final server test accuracy | Evidence |
|---|---:|---:|---|
| Vanilla RoLoRA, lr `1e-2` | 0,1,2 | `85.10 ± 2.97%` | `evidence/wandb_qnli_c50_r4_20260603/figures/proxy_plot_summary.md` |
| SVD-compensated init, lr `1e-2` | 0,1,2 | `85.70 ± 0.59%` | `results/overnight_proxy_svd_compensated_c50_r4_lr1e-2_seed{0,1,2}.log` |
| SVD-compensated + BBA, lr `1e-2` | 0 complete; 1 partial | seed 0: `86.03%`; seed 1 stopped at round 10: `76.62%` | `results/overnight_proxy_svd_compensated_bba_c50_r4_lr1e-2_seed{0,1}.log` |
| SVD-compensated + transport, lr `1e-2` | 0 | `85.36%` | `results/overnight_proxy_transport_svd_compensated_c50_r4_lr1e-2_seed0.log` |
| Orthogonal-A only, lr `1e-2` | 0 | `82.94%` | `evidence/improvement_diagnostics_20260604/proxy_orth_a_c50_r4_lr1e-2_seed0/server_metrics.csv` |
| Orthogonal-A + BBA, lr `1e-2` | 0,1 | `88.32 ± 0.22%` | `evidence/improvement_diagnostics_20260604/proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed{0,1}/server_metrics.csv` |
| Adaptive refresh + orthogonal-A, lr `1e-2` | 0,1,2 | `86.67 ± 0.31%` | `evidence/improvement_diagnostics_20260604/proxy_adaptive_refresh_orth_a_c50_r4_lr1e-2_seed{0,1,2}/server_metrics.csv` |

Interpretation:

- SVD was **already run** in actual experiments. Do not treat the toy SVD result
  as a reason to restart from scratch.
- SVD at lr `1e-2` is not the headline: it is around vanilla RoLoRA and below
  the stronger BBA/adaptive-refresh results.
- If SVD is revisited, the only remaining useful check is a lower-LR sanity run
  such as lr `5e-3`, but it is lower priority than finishing the strongest
  actual-model cells.

## Toy bank added on 2026-06-13

New direct CPU runner:

```bash
uv run python scripts/run_toy_improvement_bank.py --help
```

Main evidence:

- `evidence/toy_improvement_bank_20260613_002709_label_shard/`
- `evidence/toy_improvement_bank_20260613_003519_label_shard_confirm/`
- `evidence/toy_improvement_bank_label_shard_combined.md`

Toy label-shard result, 10 clients, one label/client, rank 16, 60 rounds:

| Toy variant | Seeds | Final accuracy |
|---|---:|---:|
| Vanilla RoLoRA | 5 | `56.08 ± 4.41%` |
| Orthogonal-A | 5 | `58.40 ± 1.44%` |
| SVD-compensated | 5 | `64.20 ± 1.99%` |
| Stale-factor stress baseline | 5 | `38.96 ± 6.83%` |
| Stale-factor + transport | 5 | `43.61 ± 6.08%` |

Interpretation:

- The toy bank is a **filter and mechanism microscope**, not a replacement for
  proxy evidence.
- It says SVD can be promising in the toy, but actual QNLI proxy evidence already
  says SVD is not currently the strongest story.
- The stale-factor transport result is a controlled stress test. In normal
  sampled-client full downlink, sampled clients receive both factors, so ordinary
  partial participation is not automatically the stale-basis failure mode.

## Recommended next actions

1. Finish/refresh the strongest actual-model story:
   - Orthogonal-A + BBA seed 2 if missing/invalid.
   - Default-init + BBA control if the Figure-6 interaction claim still matters.
2. Keep adaptive refresh + orthogonal-A as the robust 3-seed backup.
3. Only run SVD lr `5e-3` if there is spare compute and we want to close the
   initialization-ablation loop.
4. Treat partial/stale transport as a separate mechanism/stress experiment, not
   as a normal partial-participation claim yet.

## Verification for this handoff

Fresh local checks after the toy-bank changes:

```bash
uv run ruff check notebooks/mnist_fig2.py scripts/run_toy_improvement_bank.py tests/test_mnist_fig2.py
uv run pytest tests/test_mnist_fig2.py tests/test_sls_svd_lora_init.py tests/test_sls_lora_transport.py -q
uv run python -m py_compile notebooks/mnist_fig2.py scripts/run_toy_improvement_bank.py
```

Result: ruff passed, `23 passed`, and Python compilation succeeded.
