# Combined label-shard toy improvement evidence

Combines:
- `evidence/toy_improvement_bank_20260613_002709_label_shard/results.json`
- `evidence/toy_improvement_bank_20260613_003519_label_shard_confirm/results.json`

Configuration shared by both runs: MNIST label_shard, 10 clients, 1 label/client, rank 16, 60 rounds, 10 local steps, train subset 10000, test subset 2000.

| Variant | n | Final acc mean ± std | Best acc mean ± std |
|---|---:|---:|---:|
| lora | 3 | 51.17 ± 6.00 | 51.22 ± 6.04 |
| ffa | 3 | 52.88 ± 5.67 | 52.88 ± 5.67 |
| rolora | 5 | 56.08 ± 4.41 | 56.11 ± 4.44 |
| orth | 5 | 58.40 ± 1.44 | 58.40 ± 1.44 |
| orth_bba | 5 | 57.92 ± 1.16 | 57.95 ± 1.19 |
| orth_bbba | 3 | 57.65 ± 1.33 | 57.65 ± 1.33 |
| orth_bbbba | 3 | 57.52 ± 1.39 | 57.52 ± 1.39 |
| orth_transport | 3 | 57.48 ± 1.07 | 57.48 ± 1.07 |
| svd | 5 | 64.20 ± 1.99 | 64.21 ± 2.00 |
| svd_bba | 3 | 61.90 ± 3.55 | 61.90 ± 3.55 |
| partial | 3 | 51.90 ± 3.87 | 53.45 ± 3.07 |
| partial_stale | 5 | 38.96 ± 6.83 | 44.21 ± 3.25 |
| partial_stale_transport | 5 | 43.61 ± 6.08 | 47.82 ± 4.02 |

Promotion notes:

- `svd` is the strongest candidate: +8.12 pp final over vanilla RoLoRA across the 5 matched seeds.
- `orth` is a lower-risk candidate: +2.32 pp final over vanilla RoLoRA across the 5 matched seeds.
- `partial_stale_transport` is a mechanism test, not a normal partial-participation result: +4.65 pp final / +3.61 pp best over the stale-factor stress baseline across the 5 matched seeds.
