# Experiments

YAML configs that map 1:1 to runs. The training entrypoint in `code/` consumes a single config path — never command-line flag soup.

## Planned naming
```
configs/
  <method>_<dataset>_<clients>c_<rank>r_<seed>s.yaml
```
Example: `rolora_mnli_50c_4r_0s.yaml`.

## Headline cells we plan to reproduce
- RoBERTa-Large on MNLI / QQP / QNLI at 3 / 20 / 50 clients, ranks 4 and 8, ≥3 seeds.
- Local-steps ablation (Table 4) on 3 clients, 1 dataset.
- MNIST 2-layer toy (Fig. 2) for cheap sanity checking — lives in `notebooks/`, not here.

See `docs/research/deep-research-plan.md` for the full GPU-hour budget (~600–900 GPU-h on V100/A40 for the headline table).
