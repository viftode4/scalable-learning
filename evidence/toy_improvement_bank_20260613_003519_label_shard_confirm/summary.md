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
- seeds: 3, 4

## Results

| Variant | Final acc mean ± std | Best acc mean ± std | Note |
|---|---:|---:|---|
| rolora | 55.77 ± 0.82 | 55.77 ± 0.82 | vanilla RoLoRA B/A alternation |
| orth | 59.38 ± 1.18 | 59.38 ± 1.18 | orthogonal-A initialization; proposal axis 1 |
| orth_bba | 58.15 ± 1.05 | 58.22 ± 1.13 | orthogonal-A plus BBA phase pattern; tests init x schedule interaction |
| svd | 64.53 ± 0.02 | 64.53 ± 0.02 | SVD-compensated init; proposal axis 1 variant |
| partial_stale | 44.00 ± 2.80 | 47.70 ± 0.75 | stress test: sampled clients receive only active factor, exposing stale frozen factors |
| partial_stale_transport | 48.62 ± 3.08 | 52.37 ± 0.67 | basis transport under the stale-factor stress test |

## Interpretation rule

Promote a candidate to RoBERTa only if it beats vanilla RoLoRA in this toy
bank by at least about 1 pp on mean final or best accuracy, or if it is a
mechanism-specific stress test such as partial stale-basis transport.
