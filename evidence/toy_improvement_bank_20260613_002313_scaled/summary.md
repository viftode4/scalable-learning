# Toy improvement bank

Fast MNIST toy triage for RoLoRA improvement candidates. This is not a GLUE claim;
it is a cheap filter for which ideas deserve RoBERTa-base / RoBERTa-Large runs.

## Configuration

- clients: 10
- rounds: 30
- local steps: 8
- rank: 4
- train subset: 5000
- test subset: 1500
- seeds: 0, 1, 2

## Results

| Variant | Final acc mean ± std | Best acc mean ± std | Note |
|---|---:|---:|---|
| lora | 54.42 ± 0.89 | 54.42 ± 0.89 | standard LoRA; averages A and B separately |
| ffa | 48.20 ± 1.96 | 48.20 ± 1.96 | FFA-LoRA; freezes initial A and trains B |
| rolora | 46.69 ± 1.94 | 46.69 ± 1.94 | vanilla RoLoRA B/A alternation |
| orth | 46.18 ± 1.16 | 46.18 ± 1.16 | orthogonal-A initialization; proposal axis 1 |
| orth_bba | 47.71 ± 0.76 | 47.71 ± 0.76 | orthogonal-A plus BBA phase pattern; tests init x schedule interaction |
| orth_transport | 46.18 ± 1.12 | 46.18 ± 1.12 | basis transport after A-rounds |
| partial | 46.22 ± 1.44 | 46.24 ± 1.42 | ordinary partial participation with full sampled-client sync |
| partial_stale | 45.98 ± 1.58 | 45.98 ± 1.58 | stress test: sampled clients receive only active factor, exposing stale frozen factors |
| partial_stale_transport | 46.00 ± 1.64 | 46.00 ± 1.64 | basis transport under the stale-factor stress test |
| svd | 39.78 ± 5.75 | 39.78 ± 5.75 | SVD-compensated init; proposal axis 1 variant |

## Interpretation rule

Promote a candidate to RoBERTa only if it beats vanilla RoLoRA in this toy
bank by at least about 1 pp on mean final or best accuracy, or if it is a
mechanism-specific stress test such as partial stale-basis transport.
