# Notebooks

Cheap, fast exploration. Everything here is the toy reproduction of the
RoLoRA paper's §4.2 experiment (the `ReLU(xAB)·W_out` model on MNIST) plus
the comparison harness for trying RoLoRA-improvement variants against it.

Runs in minutes on a laptop (CPU / MPS / CUDA all supported, auto-detected).
**Heavier work (RoBERTa-Large / Llama) does not belong here** — use
`experiments/` + `slurm/` for that.

---

## Layout

```
notebooks/
├── README.md                  ← this file
├── mnist_fig2.py              ← entry point: paper Figure 2 reproduction
├── mnist_fig2_compare.py      ← entry point: improvement-variant comparison + ceiling overlay
└── toy/                       ← reusable building blocks (no CLI)
    ├── __init__.py            ← public API re-exports
    ├── model.py               ← PaperToyModel, MLP, LoRALinear, init_factor
    ├── data.py                ← iid_split, label_split (with repetition support)
    ├── client.py              ← local_train (LoRA+ / FedProx), evaluate
    ├── server.py              ← broadcast, average_factor, ServerMomentum
    ├── config.py              ← MethodConfig dataclass + PRESETS registry
    ├── rounds.py              ← run_method (federated), run_centralized (ceiling)
    ├── plotting.py            ← plot_curves (two-panel loss/acc overlay)
    ├── device.py              ← pick_device (CUDA → MPS → CPU)
    └── sweep.py               ← entry point: stress sweep across (clients, labels)
```

### What lives where

- **`mnist_fig2.py`** — the original Figure-2 reproduction. Runs all three
  paper methods (LoRA / FFA-LoRA / RoLoRA) at one configuration and plots
  them. Thin CLI wrapper over `toy.run_method`. Re-exports the toy package's
  public surface so legacy tests keep working with `import mnist_fig2`.
- **`mnist_fig2_compare.py`** — the comparison harness. Runs the four
  RoLoRA-improvement variants (LoRA+, orthogonal-A init, FedProx, server
  momentum) against `base_rolora`, plus the centralized non-federated
  ceiling as a dashed reference line, all overlayed on one plot.
- **`toy/sweep.py`** — the stress-grid orchestrator. Iterates over
  `(clients, labels_per_client) × seeds`, writes one PNG per cell and one
  aggregated CSV for downstream analysis.
- **`toy/`** — the package every entry point consumes. No CLI; pure
  library code. Add new variants by editing `toy/config.py:PRESETS`; add
  new splits or models by editing `toy/data.py` or `toy/model.py`.

### Available variants (`toy.config.PRESETS`)

| Key                    | What it adds on top of RoLoRA                              |
| ---------------------- | ---------------------------------------------------------- |
| `base_lora`            | (baseline) standard LoRA, no alternation                   |
| `base_ffa_lora`        | (baseline) FFA-LoRA, B-only training                       |
| `base_rolora`          | (baseline) plain RoLoRA, α=η_B/η_A=1, no prox, no momentum |
| `rolora_plus_lr`       | LoRA+ asymmetric lr (η_B = 16·η_A)                         |
| `rolora_orth_a`        | orthogonal init for A (Kaiming for B)                      |
| `rolora_prox`          | FedProx penalty on the active matrix (μ=0.01)              |
| `rolora_mom`           | Server-side Polyak momentum on aggregation (β=0.9)         |
| `rolora_kitchen_sink`  | all four of the above combined                             |
| `centralized`          | non-federated training on the union of all client data    |

The motivation for picking these four (and *not* PiSSA / DoRA / etc.) is
in `docs/deep-research-improvements.md`.

---

## How to run

All commands assume the repo root as cwd. `uv` resolves the environment.

### 1. Paper Figure 2 reproduction

```bash
# Quick smoke (CPU-friendly, finishes in ~30 s):
make mnist-smoke

# Paper-faithful run (5 clients × 2 labels, rank 1, 200 rounds):
make mnist-paper

# Default (the script's argparse defaults):
make mnist
# Or with explicit knobs:
uv run python notebooks/mnist_fig2.py \
    --clients 10 --labels-per-client 1 --rounds 100 --rank 16
```

Output → `results/mnist_fig2_<config-tag>.png` (1×2 panel: loss + accuracy
vs round, one curve per method).

### 2. Improvement-variant comparison

```bash
# Default: all 8 variants overlayed at 10 clients × 1 label, 100 rounds:
make mnist-compare

# Custom: pick which variants to overlay
uv run python notebooks/mnist_fig2_compare.py \
    --clients 20 --labels-per-client 1 \
    --variants base_rolora,rolora_plus_lr,rolora_prox \
    --rounds 50 --rank 16
```

Output → `results/mnist_fig2_compare_<tag>.{png,json}`. The PNG overlays
every variant on the same axes; the centralized ceiling is drawn dashed.
The JSON dumps `final_loss / final_acc / best_acc` per variant.

### 3. Centralized ceiling only

```bash
make mnist-ceiling
```

Same plot machinery, but only the centralized baseline runs — useful to
know how much headroom the toy task and model have before any federated
gap is even relevant.

### 4. Stress sweep across grid cells

```bash
# Default: 4 cells (5c×2, 10c×1, 20c×1, 50c×1) × 1 seed:
make mnist-stress

# Full multi-seed sweep:
uv run python notebooks/toy/sweep.py \
    --grid "5,2 10,1 20,1 50,1" \
    --seeds 0,1,2 \
    --rounds 100
```

Output → `results/sweep/c<C>_l<L>_s<S>.png` (one per cell) +
`results/sweep/summary.csv` aggregating final / best / AUC accuracy
across the whole grid.

The `(20,1)` and `(50,1)` cells shard each MNIST class across 2 and 5
client-owners respectively, stressing the federated algorithm at heavier
non-IID + smaller per-client data than the paper settings.

---

## Adding a new variant

Add one entry to `notebooks/toy/config.py:PRESETS`:

```python
"rolora_my_idea": MethodConfig(
    name="RoLoRA + my idea",
    alternation="rolora",
    lr_a=_BASE_LR,
    lr_b=_BASE_LR * 2.0,    # half-strength LoRA+
    init_a="orthogonal",
    prox_mu=0.005,
    server_momentum=0.0,
),
```

The federated loop in `rounds.run_method` dispatches on the dataclass — no
per-variant special-casing required. If your idea needs a new knob that
isn't representable as `(lr_a, lr_b, init_a, init_b, prox_mu,
server_momentum)`, extend the `MethodConfig` dataclass and add the wiring
in `rounds.run_method`. The corresponding plumbing primitives live in
`toy/client.py` (per-step) and `toy/server.py` (per-round aggregation).

Then include it in a comparison run:

```bash
uv run python notebooks/mnist_fig2_compare.py \
    --variants base_rolora,rolora_my_idea --rounds 100
```

---

## Testing

The notebooks layer is covered by four test files:

- `tests/test_mnist_fig2.py` — smoke test of `run_method` for each method.
- `tests/test_aggregation_invariants.py` — server/client invariants
  (assert_factor_identical, broadcast, average).
- `tests/test_init_conventions.py` — LoRA-on-base init convention checks.
- `tests/test_toy_components.py` — the refactor's new surface:
  label_split with repetition, init_factor variants, prox-pulls-toward-anchor,
  ServerMomentum β=0 bit-identical to average_factor, MethodConfig
  validation, run_centralized sanity.

Run them all with `make test` (or `uv run pytest -q`).
