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
| 2026-05-25 | MPS feasibility patch | `code/harness/rolora-supplement/RoLoRA-code/federatedscope/core/gpu_manager.py`, `code/harness/.../llm/trainer/trainer.py`, `results/roberta_large_feasibility_rolora.log` | Local RoBERTa-Large feasibility now runs on Apple MPS (CUDA fallback to MPS in `GPUManager.auto_choice`; `.half()` skipped on non-CUDA devices). Pre-cluster sanity path on Mac. |
| 2026-05-25 | Four QNLI reproduction configs + experiment matrix reshape | `experiments/configs/repro_qnli_c{3,20,50}_r{4,8}.yaml`, `docs/experiment-matrix.md` | Paper-cell hyperparameters pinned (4 cells × 3 methods × 3 seeds = 36 jobs/dataset). FlexLoRA omission disclosed. |
| 2026-05-25 | C2 cluster pipeline + wandb live | `slurm/repro_qnli_c20_r4_{lora,ffa_lora,rolora}.sbatch`, `scripts/{sync_to_delftblue,warm_caches}.sh`, supplement patches in `federatedscope/core/workers/{client,server}.py`, `docs/setup/delftblue.md`, `https://wandb.ai/scalable-learning-7/sls-rolora-repro` | End-to-end submission path ready: sync → warm cache (login node) → sbatch (3 modes, partition-compliant). Wandb logs `server/*` (aggregated paper number) + `client_NN/*` (per-client diagnostic). |
| 2026-05-25 | DelftBlue first-submit constraint discoveries | `slurm/repro_qnli_c20_r4_*.sbatch`, `experiments/configs/repro_qnli_c*.yaml`, `docs/setup/delftblue.md` | Recorded three cluster constraints surfaced by real `sbatch` attempts: `gpu-a100-small` caps `mem-per-cpu` at 8000 MB and `cpus-per-task` at 2; compute nodes lack outbound network for `huggingface.co`; `eval.count_flops: True` triggers CUDA-allocator pollution → `CUBLAS_STATUS_ALLOC_FAILED`. All three fixed. |
| 2026-05-27 | Trainer alternation + classifier-fix verified end-to-end | `code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/trainer/trainer.py`, `code/harness/.../federatedscope/core/workers/client.py`, `results/overnight_adamw_40.log`, `results/overnight_smoke_final.log` | Daniel's classifier-unfreeze was correct; the cluster's stuck-at-chance was an SGD-undertraining problem, not the fix. Moved alternation + `step_count++` inside the TRAIN-mode guard (was firing on val/test too, drifting `step_count` 3× per round and re-flipping `requires_grad`). Gated the wandb mech probe to client #1 so `share_local_model=True` doesn't overwrite start-of-round values with mid-round mutations. Added opt-in stdout probes (`SLS_DEBUG_PROBE=1`, `SLS_DEBUG_GRAD=1`). Per-batch grad probe proves alternation is exact: A.grad=None in B-rounds, B.grad=None in A-rounds. |
| 2026-06-03 | W&B proxy figure bundle | `scripts/plot_wandb_proxy.py`, `evidence/wandb_qnli_c50_r4_20260603/figures/` | Converts the 28-run W&B export into draft-ready proxy figures/tables: convergence curves, final/best accuracy, RoLoRA LR sweep, per-seed trajectories, and walltime/crashed-state audit. Scope is QNLI / RoBERTa-base / 50 clients / rank 4 / 20 rounds; explicitly not RoBERTa-Large Table-1 reproduction. |
| 2026-06-03 | Toy heterogeneity + orthogonal-A figure bundle | `scripts/plot_toy_heterogeneity.py`, `evidence/toy_heterogeneity_20260603/` | Copies Daniel's branch JSON results into the evidence tree and replots them for the draft: LoRA/FFA-LoRA/RoLoRA/centralized under 10 clients × one label/client, plus paired orthogonal-A RoLoRA deltas. |
| 2026-06-03 | Proposal improvement axes command-ready | `code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/model/adapter_builder.py`, `code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/trainer/{trainer.py,sls_lora_lr.py}`, `tests/test_sls_{orthogonal_lora_init,lora_lr_groups}.py`, `experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2*.yaml` | Adds `SLS_LORA_INIT=orthogonal_a`, `SLS_LORA_LR_A/B` param groups, and a FedOpt Adam proxy config. Defaults stay baseline-equivalent when env vars / FedOpt config are absent. |
| 2026-06-03 | Historical W&B offline backfill | `results/wandb/offline-run-*`; W&B project `scalable-learning-7/sls-rolora-repro` | Synced old offline W&B buffers from 2026-05-26/27 and 2026-06-03 so past local experiments are visible in the team project, not just local logs. |

## Local evidence collected

| Date | Command | Scale | Evidence | Interpretation |
|---|---|---|---|---|
| 2026-05-14 | `make mnist-paper` | MNIST, 5 clients, rank 1, 200 rounds | `results/mnist_fig2.png`; final acc: RoLoRA `0.4794`, LoRA `0.4631`, FFA-LoRA `0.3767` | Reproduces the paper's qualitative Fig. 2 ordering locally. |
| 2026-05-14 | `make supplement-smoke-all` | RoBERTa-base QNLI, 2 clients, 2 rounds, 2 local batches | `results/smoke_*.log`; all modes emit `[sls-rolora]` markers | Authors' supplement harness + three-mode patch execute locally. |
| 2026-05-14 | `make table1-pilot MODE={rolora,lora,ffa_lora}` | RoBERTa-base QNLI, 3 clients, 3 rounds, 3 local batches | `results/table1_pilot_*.log`; summarize with `make table1-pilot-summary` | Table-1-shaped pipeline works locally, but scale is too tiny for paper-comparable numbers. |
| 2026-05-14 | `make table1-medium MODE=rolora` | RoBERTa-base QNLI, 3 clients, 10 rounds, 5 local batches | `results/table1_medium_rolora.log`; summarize with `make table1-medium-summary`; final acc: test `0.511258`, val `0.504298` | Stronger single-mode local run completes; next step is `make table1-medium-all` if we can spare the runtime. |
| 2026-05-27 | `MODE=rolora TAG=adamw_40 bash scripts/run_supplement_arm.sh experiments/configs/overnight_local_qnli.yaml train.optimizer.type AdamW train.optimizer.lr 0.0005` | RoBERTa-base QNLI, 3 IID clients, 40 rounds, 20 local batches, AdamW lr 5e-4 | `results/overnight_adamw_40.log`; server-aggregated trajectory test_acc 0.51 (r1) → 0.79 (r3) → 0.85 (r11) → **0.8766 (r39)** | Disambiguates the cluster's chance-accuracy story: the model can absolutely learn QNLI on the fixed trainer; the authors' SGD lr=0.005 is dramatically undertrained. The cluster sbatch scripts inherit the SGD setup and need to be re-optimized before the next submission. |
| 2026-06-03 | `uv run python scripts/plot_wandb_proxy.py` | W&B export: QNLI, RoBERTa-base, 50 clients, rank 4, 20 rounds, 28 runs | `evidence/wandb_qnli_c50_r4_20260603/figures/proxy_plot_summary.md`; figures `proxy_a_*` through `proxy_e_*` | Draft-ready proxy baseline: RoLoRA `lr=1e-2`, seeds 0/1/2 reaches **85.10 ± 2.97%** final/best accuracy; `lr=5e-3` reaches **85.07 ± 0.90%** best accuracy. LoRA/FFA-LoRA proxy runs remain around 55-56% best and should be treated as diagnostic because full config provenance is missing. |
| 2026-06-03 | `uv run python scripts/plot_toy_heterogeneity.py` | MNIST toy, 10 clients, one label/client, 100 rounds, rank 16, 5 seeds | `evidence/toy_heterogeneity_20260603/figures/toy_plot_summary.md`; figures `toy_g_*`, `toy_h_*` | Extreme heterogeneity toy result: LoRA **62.30 ± 2.12%**, FFA-LoRA **62.51 ± 1.46%**, RoLoRA **83.89 ± 1.79%**, centralized ceiling **97.49 ± 0.15%**. Orthogonal-A RoLoRA reaches **87.19 ± 1.14%**, a **+3.30 pp** final-accuracy gain over vanilla RoLoRA. |
| 2026-06-03 | `SLS_LORA_INIT=orthogonal_a SLS_LORA_LR_A=0.005 SLS_LORA_LR_B=0.01 MODE=rolora TAG=smoke_improve_orth_ab ... smoke_supplement.yaml` | Real supplement smoke, RoBERTa-base QNLI, 2 clients, 2 rounds | `results/overnight_smoke_improve_orth_ab.log` | Harness proof for proposal axes 1 and 2: orthogonal-A initialized 24 LoRA-A matrices / zeroed 24 LoRA-B matrices; B-round optimizer groups used LoRA-B lr 0.01, A-round groups used LoRA-A lr 0.005. Not a result claim. |
| 2026-06-03 | `SLS_LORA_INIT=orthogonal_a MODE=rolora TAG=proxy_orth_a_c50_r4_lr1e-2_seed0 ... proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml seed 0` | QNLI, RoBERTa-base, 50 clients, rank 4, 20 rounds | `results/overnight_proxy_orth_a_c50_r4_lr1e-2_seed0.log`; `evidence/improvement_diagnostics_20260603/proxy_orth_a_seed0_partial/`; W&B run `wpzz95ms` | Matched orthogonal-A proxy transfer reached server test/val **0.829398 / 0.823473** at r19. It beats the earlier vanilla seed-0 control (`~0.8214`) but is currently weaker than BBA+orthogonal. |
| 2026-06-03 | `SLS_LORA_INIT=orthogonal_a SLS_PHASE_PATTERN=BBA SLS_MONITOR=1 MODE=rolora TAG=proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed0 ... proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml seed 0` | Running: QNLI, RoBERTa-base, 50 clients, rank 4, 20 target rounds | `results/overnight_proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed0.log`; `evidence/improvement_diagnostics_20260603/proxy_phase_bba_orth_a_seed0_partial/`; W&B run `8a2yoamg` | Best live improvement candidate: final server test/val **0.885411 / 0.889313** at r19. Internal monitor confirms intended phase freezing (`ΔA=0` on B rounds, `ΔB=0` on A rounds). |

## Historical local run audit

Backfilled 2026-06-03 so earlier experiments are not lost. These are local
diagnostic runs unless explicitly marked as figure/report evidence. Use them to
explain harness/optimizer decisions; do **not** present them as RoBERTa-Large
Table-1 reproduction.

W&B backfill command used:

```bash
code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/bin/wandb sync \
  --entity scalable-learning-7 --project sls-rolora-repro results/wandb/offline-run-*
```

Historical offline run IDs synced/checked: `uxzhzkz3`, `s3lbo9t2`,
`gctvkb7j`, `6z89qonj`, `pyx8ugis`, `e4fb51d1`, `s4z4osyb`, `d18oxfu9`,
`qxmvxrlq`, `a0b73cni`, `keo4ac2k`, `o3glgu5s`, `nl8oqw55`, `xz4j684t`,
`7klzzdoe`, `cjsfag8o`, `4chuqyt7`, `e4z6phxw`, `p8c0en5h`.

### 2026-05-27 optimizer/control matrix

Scope: RoBERTa-base QNLI, 3 IID clients, 40 rounds, 20 local batches. The
AdamW rows prove the fixed trainer can learn; the SGD rows explain the earlier
"chance accuracy" failure mode.

| Run | Evidence | Final server test/val | Final weighted summary | Best server test/val | Interpretation |
|---|---|---:|---:|---:|---|
| RoLoRA AdamW | `results/overnight_adamw_40.log` | `0.876625 / 0.859733` | `0.872781 / 0.870229` | `0.876625 / 0.859733` | Learns strongly after trainer alternation/classifier fix; paper-scale configs should not inherit undertrained SGD blindly. |
| RoLoRA AdamW repeat tag | `results/overnight_rolora_adamw.log` | `0.876625 / 0.859733` | `0.872781 / 0.870229` | `0.876625 / 0.859733` | Same result path under method-specific tag; keeps provenance for W&B/log matching. |
| Control/original-freeze AdamW | `results/overnight_control_originalfreeze_40.log` | `0.868753 / 0.869275` | `0.868753 / 0.869275` | `0.868753 / 0.869275` | Control is competitive locally; useful as a sanity/control row, not a committed paper claim. |
| LoRA AdamW | `results/overnight_lora_adamw.log` | `0.878272 / 0.869275` | `0.873513 / 0.875000` | `0.878272 / 0.869275` | Strong local full-LoRA baseline; compare only within this 3-client diagnostic setup. |
| FFA-LoRA AdamW | `results/overnight_ffa_lora_adamw.log` | `0.860699 / 0.862595` | `0.860699 / 0.862595` | `0.860699 / 0.862595` | Learns, but trails RoLoRA/LoRA in this diagnostic setup. |
| RoLoRA SGD | `results/overnight_rolora_sgd.log` | `0.516200 / 0.515267` | `0.494600 / 0.513359` | `0.529929 / 0.529580` | Stuck near chance; evidence that the failed cluster-style setup was optimizer/undertraining, not a broken trainer. |
| LoRA SGD | `results/overnight_lora_sgd.log` | `0.521325 / 0.519084` | `0.494600 / 0.513359` | `0.529379 / 0.535305` | Same chance-level SGD failure for LoRA. |
| FFA-LoRA SGD | `results/overnight_ffa_lora_sgd.log` | `0.519312 / 0.519084` | `0.494600 / 0.513359` | `0.527549 / 0.532443` | Same chance-level SGD failure for FFA-LoRA. |

### 2026-06-03 improvement and monitoring audit

Scope unless noted: QNLI, RoBERTa-base, 50 clients, rank 4, 20 target rounds,
RoLoRA proxy config `experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml`.

| Run | Status | Evidence | Current/final server test/val | Best server test/val | Monitor signal | Interpretation |
|---|---|---|---:|---:|---|---|
| Orthogonal-A, default AB alternation, seed 0 | 20 server rounds parsed | `results/overnight_proxy_orth_a_c50_r4_lr1e-2_seed0.log`; `evidence/improvement_diagnostics_20260603/proxy_orth_a_seed0_partial/`; W&B run `wpzz95ms` | `0.829398 / 0.823473` at r19 | `0.829398 / 0.823473` | no internal monitor rows; run started before `SLS_MONITOR=1` | Positive over the earlier vanilla seed-0 control (`~0.8214`), but not as strong as BBA+orthogonal so far. |
| BBA phase controller + orthogonal-A, seed 1 | running; launched 2026-06-04 00:04 CEST in persistent tool session `73142` | `results/overnight_proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed1.log`; PID file `results/proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed1.pid` | pending | pending | `SLS_MONITOR=1` | Replication run for seed-0 improvement claim; parse after first server aggregate lands. |
| BBA phase controller + orthogonal-A, seed 0 | complete; 20 server rounds parsed | `results/overnight_proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed0.log`; `evidence/improvement_diagnostics_20260603/proxy_phase_bba_orth_a_seed0_partial/`; W&B run `8a2yoamg` | `0.885411 / 0.889313` at r19 | `0.885411 / 0.889313` | `SLS_MONITOR=1`: B rounds show `ΔA=0, ΔB>0`; A rounds show `ΔA>0, ΔB=0` | Best current draft-critical improvement candidate: it changes only the phase schedule (`BBA`) plus committed orthogonal-A init, and already beats orthogonal default before completion. |
| BBA monitor smoke | complete smoke | `results/overnight_smoke_monitor_bba_mps.log`; `evidence/improvement_diagnostics_20260603/smoke_monitor_bba_mps/` | smoke only | smoke only | client/aggregation/internal monitor CSVs produced | Harness proof that `SLS_PHASE_PATTERN=BBA`, MPS selection, and internal/aggregation logging work before long runs. |

## Next runs (in order)

1. **Let BBA+orthogonal-A finish and refresh figures:** rerun
   `scripts/extract_supplement_metrics.py` and `scripts/plot_supplement_diagnostics.py`
   on `results/overnight_proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed0.log`.
2. **If BBA remains ahead at completion:** run BBA+orthogonal-A seeds 1 and 2,
   then plot a matched vanilla RoLoRA vs orthogonal-A vs BBA+orthogonal table/curve
   for the June-11 draft.
3. **A/B LR ablation next:** run seed 0 after the BBA decision so we can tell
   whether the gain is from initialization, phase scheduling, or asymmetric
   LoRA optimization.
   ```bash
   SLS_LORA_LR_A=0.005 SLS_LORA_LR_B=0.01 MODE=rolora TAG=proxy_ab_lr_A5e-3_B1e-2_seed0 SEED=0 \
     bash scripts/run_supplement_arm.sh experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml seed 0
   ```
4. **FedOpt fallback / ablation:** run only seed 0 unless the draft needs a
   third-axis negative/positive ablation.
   ```bash
   MODE=rolora TAG=proxy_fedopt_adam1e-1_seed0 SEED=0 \
     bash scripts/run_supplement_arm.sh experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2_fedopt_adam.yaml seed 0
   ```
5. **Draft-critical writing support:** use the generated proxy/toy figures in the June-11 draft with captions that say RoBERTa-base proxy / toy evidence, not paper-scale Table 1.
6. **Paper-scale attempt, not blocker:** after the proxy/improvement story is ledgered, submit one RoBERTa-Large QNLI attempt or ledger a clean hardware/runtime blocker.

## Rules

- Do not compare `table1_local_*` metrics directly to paper Table 1.
- Every committed claim should cite a command and a log/plot path.
- Before cluster runs, keep `make check` green.
- Every non-result setup milestone that changes the experiment path should be recorded above.
