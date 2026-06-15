# Toy improvement bank

Fast MNIST toy triage for RoLoRA improvement candidates. This is not a GLUE claim;
it is a cheap filter for which ideas deserve RoBERTa-base / RoBERTa-Large runs.

## Configuration

- clients: 10
- rounds: 60
- local steps: 10
- rank: 16
- split: label_shard
- labels per client: 1
- train subset: 10000
- test subset: 2000
- seeds: 0, 1, 2, 3, 4

## Results

| Variant | Final acc mean ± std | Best acc mean ± std | Note |
|---|---:|---:|---|
| rolora | 56.08 ± 4.41 | 56.11 ± 4.44 | vanilla RoLoRA B/A alternation |
| orth | 58.40 ± 1.44 | 58.40 ± 1.44 | orthogonal-A initialization; proposal axis 1 |
| svd | 64.20 ± 1.99 | 64.21 ± 2.00 | SVD-compensated init; proposal axis 1 variant |
| rolora_fedprox_lo | 56.07 ± 4.41 | 56.10 ± 4.44 | RoLoRA + factor-wise FedProx drift correction (mu=0.02) |
| rolora_fedprox | 56.07 ± 4.43 | 56.10 ± 4.46 | RoLoRA + factor-wise FedProx drift correction (mu=0.1) |
| rolora_fedprox_hi | 55.98 ± 4.42 | 56.03 ± 4.47 | RoLoRA + factor-wise FedProx drift correction (mu=0.5) |
| orth_fedprox | 58.37 ± 1.46 | 58.37 ± 1.46 | orthogonal-A + factor-wise FedProx; stacks best init with drift correction |

## Interpretation rule

Promote a candidate to RoBERTa only if it beats vanilla RoLoRA in this toy
bank by at least about 1 pp on mean final or best accuracy, or if it is a
mechanism-specific stress test such as partial stale-basis transport.
