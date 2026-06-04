# W&B proxy plot summary

Scope: QNLI / RoBERTa-base / 50 clients / rank 4 / 20 rounds. This is proxy evidence, not RoBERTa-Large Table-1 reproduction.

| Method | LR | n | final acc mean ± CI95 | best acc mean ± CI95 | final round(s) | state(s) |
|---|---:|---:|---:|---:|---|---|
| FFA-LoRA | `5e-3` | 1 | 50.50 ± 0.00 | 53.31 ± 0.00 | 19 | crashed |
| FFA-LoRA | `1e-2` | 3 | 51.95 ± 1.01 | 54.88 ± 2.32 | 19,19,19 | crashed |
| FFA-LoRA | `2e-2` | 3 | 51.65 ± 3.75 | 55.64 ± 1.66 | 19,19,19 | crashed |
| LoRA | `5e-3` | 1 | 50.32 ± 0.00 | 53.40 ± 0.00 | 19 | crashed |
| LoRA | `1e-2` | 3 | 51.89 ± 1.20 | 54.88 ± 2.34 | 19,19,19 | crashed |
| LoRA | `2e-2` | 3 | 51.62 ± 3.60 | 55.91 ± 2.28 | 19,19,19 | crashed |
| RoLoRA | `5e-4` | 1 | 47.87 ± 0.00 | 48.32 ± 0.00 | 19 | crashed |
| RoLoRA | `1e-3` | 1 | 51.95 ± 0.00 | 51.95 ± 0.00 | 19 | crashed |
| RoLoRA | `2e-3` | 1 | 56.12 ± 0.00 | 56.12 ± 0.00 | 19 | crashed |
| RoLoRA | `5e-3` | 3 | 81.47 ± 7.23 | 85.07 ± 0.90 | 19,19,19 | crashed |
| RoLoRA | `1e-2` | 3 | 85.10 ± 2.97 | 85.10 ± 2.97 | 19,19,19 | crashed |
| RoLoRA | `2e-2` | 3 | 81.20 ± 9.39 | 83.23 ± 5.95 | 19,19,19 | crashed |
| RoLoRA | `5e-2` | 1 | 63.55 ± 0.00 | 77.15 ± 0.00 | 19 | crashed |
| RoLoRA | `1e-1` | 1 | 47.96 ± 0.00 | 52.58 ± 0.00 | 19 | crashed |

## Figure files

- `proxy_a_server_accuracy_convergence.{png,pdf}` — selected mean ± CI95 convergence curves.
- `proxy_b_final_best_by_method_lr.{png,pdf}` — final/best accuracy summary by method/LR.
- `proxy_c_rolora_lr_sweep.{png,pdf}` — RoLoRA LR sweep used to choose the proxy control.
- `proxy_d_rolora_per_seed_trajectories.{png,pdf}` — per-seed RoLoRA trajectories for lr=1e-2 and 5e-3.
- `proxy_e_walltime_crash_audit.{png,pdf}` — runtime/status audit explaining crashed labels.
