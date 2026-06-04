# Supplement metrics summary

- server rows: 0
- client eval rows: 4
- client train rows: 4
- phase markers: 4 raw, 2 unique rounds
- internal monitor rows: 12
- aggregation monitor rows: 4


## Rounds seen without server aggregate yet

| round | phase | client_test_mean | train_acc_mean | train_n |
|---:|:---:|---:|---:|---:|
| 0 | B |  | 0.500000 | 2 |
| 1 | B | 0.494415 | 0.500000 | 2 |
| 2 | A | 0.494415 |  |  |

## Internal monitor summary

Local-update source: `client_post_train` rows.

| round | phase | local_n | mean_||BA|| | mean_ΔA | mean_ΔB | mean_Δclassifier | mean_client_drift_ΔA | mean_client_drift_ΔB | aggregate_ΔA | aggregate_ΔB |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | B | 1 |  | 0.000000 | 0.001826 | 0.013546 | 0.000000 | 0.000936 | 0.000000 | 0.000936 |
| 1 | B | 1 |  | 0.000000 | 0.002485 | 0.016100 | 0.000000 | 0.002299 | 0.000000 | 0.002299 |
