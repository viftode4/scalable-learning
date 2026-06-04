# Supplement metrics summary

- server rows: 0
- client eval rows: 0
- client train rows: 9
- phase markers: 10 raw, 1 unique rounds
- internal monitor rows: 19
- aggregation monitor rows: 0


## Rounds seen without server aggregate yet

| round | phase | client_test_mean | train_acc_mean | train_n |
|---:|:---:|---:|---:|---:|
| 0 | B |  | 0.498958 | 9 |

## Internal monitor summary

Local-update source: `fit_end` rows.

| round | phase | local_n | mean_||BA|| | mean_ΔA | mean_ΔB | mean_Δclassifier | mean_client_drift_ΔA | mean_client_drift_ΔB | aggregate_ΔA | aggregate_ΔB |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | B | 9 | 0.014386 | 0.000000 | 0.014386 | 0.030360 |  |  |  |  |
