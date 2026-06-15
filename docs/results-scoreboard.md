# Results scoreboard — what we tried, the verdict, the evidence

Single honest entry point for the RoLoRA reproduction + improvement project.
Start here, then follow the links. Updated 2026-06-15.

The one-line story: **reproduction is on track but cluster-pending; on the
improvement side, every IID experiment is flat and the only thing that moves is
better initialisation under heterogeneity.** Several "obvious" improvement ideas
turned out to be closed by the paper's own results or theory — those are
recorded here so we don't re-run them.

## How to read this

- **Toy** = MNIST 2-layer model, fast CPU filter (`notebooks/toy/`,
  `mnist_fig2_compare.py`, `toy/sweep.py`). Cheap triage, *not* a paper claim.
- **Proxy** = RoBERTa-base / QNLI / 50 clients / rank 4 / 20 rounds on the
  authors' supplement harness. Real model, sub-Table-1 scale. Vanilla RoLoRA
  baseline here is **0.851 ± 0.030** (seed envelope ~0.82–0.88), so a real
  improvement has to clear that noise across ≥3 seeds.
- **Paper-scale** = RoBERTa-Large Table 1. Cluster-only; see the reproduction
  row below.

## Reproduction

| Item | Status | Evidence |
|---|---|---|
| Supplement harness runs locally (3 modes) | ✅ | `make supplement-smoke-all`; `experiments/ledger/README.md` |
| Optimizer audit (shipped SGD lr=0.005 → chance; AdamW lr=5e-4 → 0.86–0.88) | ✅ documented | `docs/decisions/0006-supplement-reproducibility-gap.md`, README banner |
| MNIST Figure-2 ordering (RoLoRA > LoRA > FFA-LoRA) | ✅ | `make mnist-paper`; `notebooks/toy/` |
| Cluster Table-1 reproduction (RoBERTa-Large) | ⏳ pending | corrected `slurm/repro_qnli_*`; first clean C2 cell is the open gate |

## Improvement experiments

Verdict legend: ✅ helps · ➖ neutral / within noise · ❌ ruled out · ⏳ pending.

| Idea | Where it lives | Toy | Proxy (real model) | Verdict |
|---|---|---|---|---|
| **Orthogonal-A init** | `toy` preset `rolora_orth_a`; supplement `SLS_LORA_INIT=orthogonal_a` | **+3.3 pp** under heterogeneity (87.2 vs 83.9, 5 seeds) | ~0.829 vs 0.821, within noise | ✅ on heterogeneity / ➖ IID — our **best signal** |
| **SVD-compensated init (PiSSA-style)** | `SLS_LORA_INIT=svd_compensated` | strong on the rank-16 bank (64.2 vs 56.1) | 0.852–0.865, ➖ neutral | ➖ does not transfer to the real-model proxy |
| **LoRA+ (asymmetric A/B LR)** | preset `rolora_plus_lr`; `SLS_LORA_LR_A/B` | λ=2 → 0.848 vs 0.844, marginal | not run — paper Fig 6 already shows balanced LR wins on the real model | ❌ closed by the paper's own Fig 6 |
| **Phase schedule (BBA etc.)** | `SLS_PHASE_PATTERN` | roughly matches orth-A, no extra gain | BBA+orth-A 0.885 vs vanilla 0.851 — an **init×schedule interaction** Fig 6 never tests | ➖ as a standalone knob; the interaction is a finding *about* the paper, not an improvement |
| **Basis transport** (re-express B after an A-round) | supplement `SLS_LORA_TRANSPORT=ls`, `sls_lora_transport.py` | pruned from the toy (non-winner) | ~0.878, one seed, ➖ | ➖ neutral on IID; built for a problem IID doesn't have |
| **Factor-wise drift correction (FedProx)** | preset `rolora_prox`; tested independently this session | **flat**: 56.07 vs 56.08, inside ±4.4 noise | not run | ➖ FedProx is the weak corrector; SCAFFOLD-style control variates are the untried stronger version |
| **Server momentum (FedAvgM)** | preset `rolora_mom` | implemented, multi-seed comparison not yet recorded | not run | ⏳ pending a clean toy/sweep run |
| **Partial participation** | n/a | — | — | ❌ ruled out by RoLoRA's exactness argument (Eqs 3-4 survive client sampling) — see `docs/decisions/0007-partial-participation-ruled-out.md` |

### The shape of the result

- **IID is dead ground.** Init tricks, schedules, transport, and drift
  correction are all flat on the IID proxy. RoLoRA is already near-optimal there.
- **Heterogeneity is where things move**, and the lever that moves them is
  **initialisation quality** (orth-A +3.3 pp). This matches the paper's own
  analysis (FFA-LoRA's error scales with A's init angle).
- **Why some ideas are closed, not just untested:** Fig 6 already ablates
  asymmetric A/B LR and B-prioritised schedules (balanced wins); the exactness
  argument already covers partial participation. Reasoning these out on paper
  saved real compute.

## Where everything lives

| Thing | Path |
|---|---|
| Toy harness (canonical) | `notebooks/toy/` · `notebooks/mnist_fig2_compare.py` · `notebooks/toy/sweep.py` |
| Toy variants registry | `notebooks/toy/config.py` (`PRESETS`) |
| Supplement harness + `SLS_*` switches | `code/harness/rolora-supplement/RoLoRA-code/federatedscope/` |
| Experiment configs | `experiments/configs/` |
| Committed results (CSVs, figures, JSON) | `evidence/`, `results_extra/` (toy heterogeneity) |
| Scratch run logs (gitignored) | `results/` |
| Per-idea reasoning records | `.plans/`, `docs/decisions/` |
| Dense run log | `experiments/ledger/README.md` |
| Improvement idea banks | `docs/current-roadmap.md`, `docs/deep-research-improvements.md` |

## Open / next

1. A clean toy/sweep run of `rolora_mom` and `rolora_kitchen_sink` to close the
   momentum row (`uv run python -m toy.sweep --variants base_rolora,rolora_mom,rolora_kitchen_sink`).
2. The first clean cluster Table-1 cell (corrected AdamW + round count).
3. If a toy lever clears the bar, the matched RoBERTa-base proxy confirmation
   (≥3 seeds, must beat 0.851 ± 0.030).

## Reproduce the toy comparison

```bash
uv run python -m toy.sweep --seeds 0,1,2          # multi-seed grid → results/*.csv
uv run python notebooks/mnist_fig2_compare.py \
  --clients 10 --labels-per-client 1              # single-seed overlay plot
```
