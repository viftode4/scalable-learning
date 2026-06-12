# Toy improvement bank

Fast MNIST toy triage for RoLoRA improvement candidates. This is not a GLUE claim;
it is a cheap filter for which ideas deserve RoBERTa-base / RoBERTa-Large runs.

## Configuration

- clients: 4
- rounds: 4
- local steps: 2
- rank: 2
- train subset: 512
- test subset: 256
- seeds: 0

## Results

| Variant | Final acc mean ± std | Best acc mean ± std | Note |
|---|---:|---:|---|
| rolora | 10.55 ± 0.00 | 10.55 ± 0.00 | vanilla RoLoRA B/A alternation |
| orth_bba | 9.38 ± 0.00 | 9.38 ± 0.00 | orthogonal-A plus BBA phase pattern; tests init x schedule interaction |
| partial_stale_transport | 9.38 ± 0.00 | 9.38 ± 0.00 | basis transport under the stale-factor stress test |

## Interpretation rule

Promote a candidate to RoBERTa only if it beats vanilla RoLoRA in this toy
bank by at least about 1 pp on mean final or best accuracy, or if it is a
mechanism-specific stress test such as partial stale-basis transport.
