# `runs/` — the shared run registry

**One committed place for every real run, so teammates get them with `git pull`.**
No more digging through local `results/` / `exp/` scratch or asking who has which log.

Start at **[`REGISTRY.md`](REGISTRY.md)** — the verified, single-source-of-truth
table of every run (final/best accuracy, status, source), generated straight from
the primary artifacts below.

## Layout

```
runs/
  REGISTRY.md                    verified aggregate of everything (the index)
  proxy/<run>/                   QNLI · RoBERTa-base · 50 clients · rank 4 · 20 rounds
    run.log                      the raw FederatedScope server log
    server_metrics.csv           per-round server test/val acc (round,test_acc,val_acc)
    config.yaml                  the resolved config the run used
    meta.json                    final/best acc, rounds, status, git_sha, init/schedule switches
  audit/<run>/                   FedAvg optimizer audit (ADR 0006): SGD-chance vs AdamW
    eval_results.log + meta.json
  baselines/                     vanilla RoLoRA / LoRA / FFA-LoRA learning-rate sweep
    server_history.csv           the W&B export (all seeds × LRs)
    meta.json                    grouped final-acc means (± CI95)
  toy/                           canonical MNIST triage runs (rank 16 · 60 rounds · ≥3 seeds)
    <run>.json
```

## Refresh it

```bash
make promote-runs          # copy canonical artifacts into runs/ + rewrite REGISTRY.md
# or, registry only (no copying):
make results-aggregate
```

`promote` is idempotent — re-run it after any new run. It reads the primary
sources (logs in `results/`, server CSVs in `evidence/`, the W&B export, FedAvg
eval logs, toy JSON), so add a run the normal way and then promote it.

## What is deliberately *not* here

Raw local scratch stays out of git and lives only on the machine that ran it:

- `results/` — full run logs incl. the ~109 MB `results/wandb/` dir, pids, queue logs
- `exp/` — FederatedScope auto-output trees
- `checkpoints/`, `*.ckpt`, `*.pt` — model weights

`runs/` carries only the curated, shareable evidence (~28 MB): the canonical log,
its parsed metrics, the config, and meta — enough to cite, plot, and reproduce
without shipping gigabytes.

## Conventions

`REGISTRY.md` reports proxy/sweep means as **mean ± CI95** (`1.96·s/√n`); toy means
as **± sample std**. The canonical metric is the final aggregation round's server
`Results_weighted_avg.test_acc`. See the header of `REGISTRY.md` for the full note
and how this reconciles with older docs.
