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


## Local evidence suites

Use `make local-smoke` for a fast setup check and `make full-local` for the strongest laptop-feasible run before moving to cluster experiments. `full-local` runs first-party tests/lint, the 200-round MNIST Figure-2 sanity check, and all three patched supplement modes on the tiny RoBERTa-base config.

Use `make table1-pilot MODE=rolora` or `make table1-pilot-all` for the next local rung: QNLI, RoBERTa-base, 3 clients, 3 communication rounds, and 3 local batches. This proves the Table-1-shaped pipeline locally without claiming paper-comparable RoBERTa-Large numbers.
