# Toy improvement bank

Fast MNIST toy triage for RoLoRA improvement candidates. This is not a GLUE claim;
it is a cheap filter for which ideas deserve RoBERTa-base / RoBERTa-Large runs.

## Configuration

- clients: 10
- rounds: 20
- local steps: 5
- rank: 4
- train subset: 3000
- test subset: 1000
- seeds: 0, 1

## Results

| Variant | Final acc mean ± std | Best acc mean ± std | Note |
|---|---:|---:|---|
| lora | 44.15 ± 0.15 | 44.15 ± 0.15 | standard LoRA; averages A and B separately |
| ffa | 41.10 ± 1.10 | 41.10 ± 1.10 | FFA-LoRA; freezes initial A and trains B |
| rolora | 38.75 ± 0.85 | 38.75 ± 0.85 | vanilla RoLoRA B/A alternation |
| orth | 11.70 ± 1.00 | 11.75 ± 1.05 | orthogonal-A initialization; proposal axis 1 |
| orth_bba | 11.95 ± 1.05 | 11.95 ± 1.05 | orthogonal-A plus BBA phase pattern; tests init x schedule interaction |
| orth_transport | 11.70 ± 1.00 | 11.75 ± 1.05 | basis transport after A-rounds |
| partial | 11.80 ± 1.10 | 11.80 ± 1.10 | ordinary partial participation with full sampled-client sync |
| partial_stale | 11.80 ± 1.10 | 11.80 ± 1.10 | stress test: sampled clients receive only active factor, exposing stale frozen factors |
| partial_stale_transport | 11.80 ± 1.10 | 11.80 ± 1.10 | basis transport under the stale-factor stress test |
| svd | 22.60 ± 0.70 | 22.60 ± 0.70 | SVD-compensated init; proposal axis 1 variant |

## Interpretation rule

Promote a candidate to RoBERTa only if it beats vanilla RoLoRA in this toy
bank by at least about 1 pp on mean final or best accuracy, or if it is a
mechanism-specific stress test such as partial stale-basis transport.
