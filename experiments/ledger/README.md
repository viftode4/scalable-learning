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

## Local evidence collected

| Date | Command | Scale | Evidence | Interpretation |
|---|---|---|---|---|
| 2026-05-14 | `make mnist-paper` | MNIST, 5 clients, rank 1, 200 rounds | `results/mnist_fig2.png`; final acc: RoLoRA `0.4794`, LoRA `0.4631`, FFA-LoRA `0.3767` | Reproduces the paper's qualitative Fig. 2 ordering locally. |
| 2026-05-14 | `make supplement-smoke-all` | RoBERTa-base QNLI, 2 clients, 2 rounds, 2 local batches | `results/smoke_*.log`; all modes emit `[sls-rolora]` markers | Authors' supplement harness + three-mode patch execute locally. |
| 2026-05-14 | `make table1-pilot MODE={rolora,lora,ffa_lora}` | RoBERTa-base QNLI, 3 clients, 3 rounds, 3 local batches | `results/table1_pilot_*.log`; summarize with `make table1-pilot-summary` | Table-1-shaped pipeline works locally, but scale is too tiny for paper-comparable numbers. |
| 2026-05-14 | `make table1-medium MODE=rolora` | RoBERTa-base QNLI, 3 clients, 10 rounds, 5 local batches | `results/table1_medium_rolora.log`; summarize with `make table1-medium-summary`; final acc: test `0.511258`, val `0.504298` | Stronger single-mode local run completes; next step is `make table1-medium-all` if we can spare the runtime. |

## Next runs (in order)

1. **First cluster submission (immediate next):**
   ```bash
   # Laptop:
   bash scripts/sync_to_delftblue.sh <netid>
   # Cluster (login node, one-time):
   bash scripts/warm_caches.sh        # ~5 min, no Ctrl-C
   # Cluster (the actual training job):
   sbatch slurm/repro_qnli_c20_r4_rolora.sbatch
   ```
   Watch `slurm_logs/sls-c20-r4-rolora-<jobid>.out` and `https://wandb.ai/scalable-learning-7/sls-rolora-repro`. Ledger result here whether pass or fail.
2. **If (1) passes**, submit the other two C2 modes (`lora`, `ffa_lora`) and seeds 1, 2 for each (9 jobs total for C2).
3. **Copy the C2 sbatch template** into C1 (3 clients), C3 (50 clients r=4), C4 (50 clients r=8) — 9 more sbatch files. Only `CONFIG_PATH`, `--job-name`, `WANDB_RUN_GROUP`, `WANDB_TAGS` need to change.
4. **Author the post-hoc tools** (`scripts/aggregate_seeds.py` for mean±std; `scripts/plot_convergence_curves.py` for Figure-3-style panel from `exp/*/sub_exp_*/eval_results.log`) once data starts landing.
5. **Optional local fill-in**: `make table1-medium-all` is still available if you want a RoBERTa-base baseline at the local rung, but it's no longer on the critical path.
6. **After QNLI is fully ledgered**, write `mnli2json.py` and submit the MNLI expansion (M3-M6 in `docs/experiment-matrix.md`).

## Rules

- Do not compare `table1_local_*` metrics directly to paper Table 1.
- Every committed claim should cite a command and a log/plot path.
- Before cluster runs, keep `make check` green.
- Every non-result setup milestone that changes the experiment path should be recorded above.
