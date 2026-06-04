# June 11 draft results brief — 2026-06-03

Use this as the report-writing source of truth until new runs complete.

## What we can already write

### 1. Reproducibility audit

The released OpenReview supplement is runnable after harness fixes, but it is
not a clean one-command reproduction of the RoLoRA paper numbers. The important
points for the draft are:

- The shipped optimizer/LR recipe undertrains QNLI; local AdamW evidence learns
  while the shipped SGD-style runs stay near chance.
- The trainer had an undocumented classifier-freeze behavior; our control shows
  it is a code-quality/reproducibility issue, not the sole cause of chance-level
  runs.
- Current paper-scale RoBERTa-Large evidence is still incomplete, so all proxy
  figures must be labelled as proxy evidence.

### 2. Existing W&B proxy baseline

Source: `evidence/wandb_qnli_c50_r4_20260603/figures/proxy_plot_summary.md`.

Scope: QNLI / RoBERTa-base / 50 clients / rank 4 / 20 rounds. This is **not**
RoBERTa-Large Table 1 reproduction.

Draft-safe numbers:

- RoLoRA `lr=1e-2`, seeds 0/1/2: final/best accuracy **85.10 ± 2.97%**.
- RoLoRA `lr=5e-3`, seeds 0/1/2: best accuracy **85.07 ± 0.90%**.
- LoRA/FFA-LoRA visible proxy runs remain around **55–56% best**, but because
  full config provenance is incomplete, frame them as diagnostic rather than a
  strong negative paper-scale claim.

Use these figures:

- `proxy_a_server_accuracy_convergence.png`
- `proxy_b_final_best_by_method_lr.png`
- `proxy_c_rolora_lr_sweep.png`
- `proxy_d_rolora_per_seed_trajectories.png`
- `proxy_e_walltime_crash_audit.png`

### 3. Existing toy heterogeneity + improvement signal

Source: `evidence/toy_heterogeneity_20260603/figures/toy_plot_summary.md`.

Scope: MNIST toy model, 10 clients, one label/client, rank 16, 100 rounds,
5 seeds.

Draft-safe numbers:

- LoRA: **62.30 ± 2.12%** final accuracy.
- FFA-LoRA: **62.51 ± 1.46%** final accuracy.
- RoLoRA: **83.89 ± 1.79%** final accuracy.
- Centralized ceiling: **97.49 ± 0.15%** final accuracy.
- RoLoRA + orthogonal-A init: **87.19 ± 1.14%**, a **+3.30 pp** gain over
  vanilla RoLoRA in this toy setting.

Use these figures:

- `toy_g_heterogeneous_baselines_curves.png`
- `toy_g_heterogeneous_baselines_final_acc.png`
- `toy_h_orthogonal_a_paired_delta.png`

Main wording: orthogonal-A has a multi-seed toy signal and is therefore the
first proposal-compatible improvement to transfer to the RoBERTa-base proxy.
Do not claim GLUE improvement until the matched proxy run completes.

## Improvement work now

All three proposal axes now have a concrete smallest-run path:

1. **Improved initialization** — implemented as `SLS_LORA_INIT=orthogonal_a`.
   - Real smoke passed: `results/overnight_smoke_improve_orth_ab.log`.
   - Draft-relevant proxy run started: `results/overnight_proxy_orth_a_c50_r4_lr1e-2_seed0.log`.
   - W&B group: `scalable-learning-7/sls-rolora-repro`, group
     `qnli_c50_r4_improvements`.

2. **Separate A/B learning rates** — implemented as env-gated param groups.
   - `SLS_LORA_LR_A=0.005 SLS_LORA_LR_B=0.01` is the first conservative probe.
   - Default behavior is unchanged when env vars are absent.

3. **Adaptive server optimization** — use FederatedScope `FedOptAggregator`.
   - Config ready: `experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2_fedopt_adam.yaml`.
   - First conservative probe uses server Adam lr `0.1`.

## Run order for the draft

1. Finish/monitor orthogonal-A proxy seed 0.
2. If seed 0 is at least competitive with vanilla seed 0 (`0.8214` final/best),
   run orthogonal-A seeds 1 and 2 and plot a matched vanilla-vs-orthogonal curve.
3. If orthogonal-A is clearly worse, run the A/B LR seed-0 probe next.
4. Run FedOpt seed 0 only after either orthogonal-A or A/B LR gives a usable
   direction, unless we need a quick negative/ablation table.

## Caption rule

Every proxy figure caption must include:

> QNLI / RoBERTa-base / 50 clients / rank 4 / 20 rounds; proxy evidence, not
> RoBERTa-Large Table-1 reproduction.

## Live improvement update — 2026-06-03 evening

The first local M4/MPS transfer run for `orthogonal_a` on the QNLI C50/r4 proxy is in progress. Partial diagnostics support the phase-controller hypothesis: B phases are producing the useful gains, while the following A phase can temporarily erase them. Current extracted rows are:

| round | phase | server test acc | delta |
|---:|:---:|---:|---:|
| 0 | B | 0.509793 | — |
| 1 | A | 0.509061 | -0.000732 |
| 2 | B | 0.563793 | +0.054732 |
| 3 | A | 0.506864 | -0.056929 |

Round 4 is a B phase and has a pending server aggregate, but the complete client-eval mean is already about `0.722724`. This justifies testing a lightweight phase-controller layer over RoLoRA rather than changing the whole method: `SLS_PHASE_PATTERN=BBA` (B/B/A repeating) is queued as the next seed-0 run with `orthogonal_a` and explicit `SLS_DEVICE=mps`.

## Live monitor update — 2026-06-03 night

Instrumentation is now sufficient for draft-time diagnosis, not just final accuracy. For monitored runs (`SLS_MONITOR=1`) the harness logs factor norms, active-factor update norms, and aggregation drift/update norms. The active phase-controller probe is:

- `SLS_LORA_INIT=orthogonal_a SLS_PHASE_PATTERN=BBA SLS_DEVICE=mps SLS_MONITOR=1 MODE=rolora`
- log: `results/overnight_proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed0.log`
- W&B: `https://wandb.ai/scalable-learning-7/sls-rolora-repro/runs/8a2yoamg`
- plots/CSVs: `evidence/improvement_diagnostics_20260603/proxy_phase_bba_orth_a_seed0_partial/`

The first round is still before server aggregation, but the monitor table already confirms the intended B-only behavior: mean `ΔA=0.000000`, mean `ΔB=0.017690`, and mean `Δclassifier=0.030826` over the first 30 local client fits. Use `supplement_internal_monitors.png` for the mechanism plot and `supplement_diagnostics_curve.png` once server eval rows arrive.
