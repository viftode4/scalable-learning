# Toy heterogeneity plot summary

Scope: MNIST toy model, 10 clients, one label per client, label split, 100 rounds, rank 16, 5 seeds.
Source: `origin/fix-rolora:results_extra/{baselines_n5_r100_log5.json,orth_a_n5_r100_log5.json}` copied into this evidence directory.

| Variant | Final acc mean ± CI95 | Best acc mean ± CI95 |
|---|---:|---:|
| LoRA | 62.30 ± 2.12 | 62.30 ± 2.12 |
| FFA-LoRA | 62.51 ± 1.46 | 62.53 ± 1.47 |
| RoLoRA | 83.89 ± 1.79 | 83.97 ± 1.81 |
| Centralized (non-federated ceiling) | 97.49 ± 0.15 | 97.52 ± 0.14 |
| RoLoRA + orthogonal-A init | 87.19 ± 1.14 | 87.70 ± 0.72 |

Orthogonal-A final-accuracy gain over vanilla RoLoRA: **3.30 percentage points**.

## Figure files

- `toy_g_heterogeneous_baselines_curves.{png,pdf}` — loss/accuracy curves with CI95 bands.
- `toy_g_heterogeneous_baselines_final_acc.{png,pdf}` — final accuracy summary.
- `toy_h_orthogonal_a_paired_delta.{png,pdf}` — paired seed deltas for orthogonal-A.
