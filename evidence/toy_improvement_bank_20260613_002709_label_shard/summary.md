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
- seeds: 0, 1, 2

## Results

| Variant | Final acc mean ± std | Best acc mean ± std | Note |
|---|---:|---:|---|
| lora | 51.17 ± 6.00 | 51.22 ± 6.04 | standard LoRA; averages A and B separately |
| ffa | 52.88 ± 5.67 | 52.88 ± 5.67 | FFA-LoRA; freezes initial A and trains B |
| rolora | 56.28 ± 5.65 | 56.33 ± 5.68 | vanilla RoLoRA B/A alternation |
| orth | 57.75 ± 1.21 | 57.75 ± 1.21 | orthogonal-A initialization; proposal axis 1 |
| orth_bba | 57.77 ± 1.20 | 57.77 ± 1.20 | orthogonal-A plus BBA phase pattern; tests init x schedule interaction |
| orth_bbba | 57.65 ± 1.33 | 57.65 ± 1.33 | orthogonal-A plus BBBA phase pattern; longer B warm-up before A motion |
| orth_bbbba | 57.52 ± 1.39 | 57.52 ± 1.39 | orthogonal-A plus BBBBA phase pattern; aggressive B warm-up |
| orth_transport | 57.48 ± 1.07 | 57.48 ± 1.07 | basis transport after A-rounds |
| partial | 51.90 ± 3.87 | 53.45 ± 3.07 | ordinary partial participation with full sampled-client sync |
| partial_stale | 35.60 ± 6.66 | 41.88 ± 1.91 | stress test: sampled clients receive only active factor, exposing stale frozen factors |
| partial_stale_transport | 40.27 ± 5.23 | 44.78 ± 1.88 | basis transport under the stale-factor stress test |
| svd | 63.98 ± 2.55 | 64.00 ± 2.56 | SVD-compensated init; proposal axis 1 variant |
| svd_bba | 61.90 ± 3.55 | 61.90 ± 3.55 | SVD-compensated init plus BBA; tests whether compensated bases need slower A updates |

## Interpretation rule

Promote a candidate to RoBERTa only if it beats vanilla RoLoRA in this toy
bank by at least about 1 pp on mean final or best accuracy, or if it is a
mechanism-specific stress test such as partial stale-basis transport.
