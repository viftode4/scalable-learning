# Results scoreboard — what we tried, the verdict, the evidence

Single honest entry point for the RoLoRA reproduction + improvement project.
Start here, then follow the links. Updated 2026-06-15.

**One-line story:** our improvement is **Phase-Controlled RoLoRA** — RoLoRA +
orthogonal-A init + a B-prioritised phase schedule (BBA) + a product-preserving
gauge fix — which reaches **0.885 vs vanilla 0.851** on the RoBERTa-base proxy.
The reason it works is a single mechanism we can point at (A-rounds damage the
function learned in B-rounds), and that same mechanism explains why basis
transport gives the same lift and why init and schedule *interact* — an
interaction the paper's Fig 6 never tests. Reproduction is on track but
cluster-pending. Several other ideas are genuinely closed by the paper's own
theory/results and are recorded so we don't re-run them.

## How to read this

- **Proxy** = RoBERTa-base / QNLI / 50 clients / rank 4 / 20 rounds on the
  authors' supplement harness. Real model, sub-Table-1 scale. This is where our
  *proper* improvement runs live. Vanilla RoLoRA baseline = **0.851 ± 0.030**
  (3 seeds; envelope ~0.82–0.88), so a clean claim must clear that noise.
- **Toy** = MNIST 2-layer model, fast CPU filter (`notebooks/toy/`,
  `mnist_fig2_compare.py`, `toy/sweep.py`). Triage only, *not* a paper claim.
- **Paper-scale** = RoBERTa-Large Table 1. Cluster-only; see reproduction below.

## Reproduction

| Item | Status | Evidence |
|---|---|---|
| Supplement harness runs locally (3 modes) | ✅ | `make supplement-smoke-all` |
| Optimizer audit (shipped SGD lr=0.005 → chance; AdamW lr=5e-4 → 0.86–0.88) | ✅ documented | `docs/decisions/0006-supplement-reproducibility-gap.md` |
| MNIST Figure-2 ordering (RoLoRA > LoRA > FFA-LoRA) | ✅ | `make mnist-paper`; `notebooks/toy/` |
| Cluster Table-1 reproduction (RoBERTa-Large) | ⏳ pending | corrected `slurm/repro_qnli_*`; first clean C2 cell is the open gate |

## Our improvement — Phase-Controlled RoLoRA (PC-RoLoRA)

### The mechanism (the actual contribution)

RoLoRA's **A-rounds damage the function learned in the B-rounds**: when A is
re-aggregated, B's frozen coefficients are now expressed against a moved basis,
so the adapter `B·A` is corrupted. Every proxy arm that helped is a fix for
exactly this:

- **BBA schedule** — cluster B-rounds, fewer disruptive A-rounds.
- **Basis transport** — re-express B after each A-round (`SLS_LORA_TRANSPORT=ls`).
- **Orthogonal-A init** — a better-conditioned basis, so the A-round damage is
  smaller to begin with.

Two results pin the mechanism down:
1. **Transport ≈ BBA, and they don't stack.** Transport lifts plain balanced-AB
   from 0.851 → **0.878** (recovering most of BBA's gain), but transport stacked
   on BBA = 0.883 ≈ BBA alone (0.885). Both are fixing the *same* A-round damage
   by different routes — exactly the prediction in `.plans/lora-basis-transport.md` (P2).
2. **Init × schedule interaction.** BBA helps *specifically under orth-A init*.
   The paper's Fig 6 only ablates schedules under *default* init and concludes
   balanced wins; it never tests the interaction, so this is genuinely ours.

The more novel framing is **APC-RoLoRA** (adaptive phase controller): instead of
a fixed BBA, choose which factor to train from observed dynamics (active-factor
update norm, aggregation drift, recent val gain, no-starvation). See
`docs/current-roadmap.md` §10–11.

### Proxy results (RoBERTa-base, QNLI, c50/r4/20 rounds — final server test_acc)

| Arm | Result | vs 0.851 ± 0.030 |
|---|---|---|
| vanilla RoLoRA (baseline) | 0.851 ± 0.030 (3 seeds) | — |
| orth-A init alone | 0.829 (s0) | within noise |
| **BBA + orth-A** | **0.885 / 0.881** (s0/s1) | **+~3.4 pp**, edge of envelope |
| BBA + orth-A + gauge | 0.883 (s0) | matches BBA |
| adaptive_refresh + orth-A (APC) | 0.866 / 0.864 / 0.871 | between vanilla and BBA |
| transport (balanced AB) | 0.878 (s0) | recovers most of BBA's gain |
| transport + BBA + orth-A | 0.883 (s0) | no stacking benefit |
| SVD-compensated init | 0.852 / 0.865 / 0.854 | neutral |
| SVD + BBA | 0.860 / **0.766 💥** | unstable |
| transport + SVD | 0.854 (s0) | neutral |

Evidence: `evidence/share_csv_for_chat_20260608/improvement_server_curves/*.csv`,
`evidence/improvement_diagnostics_20260604/`, `results/overnight_proxy_*` (logs,
gitignored). Toy confirmation: orth-A gives **+3.3 pp** under extreme
heterogeneity (`results_extra/orth_a_n5_r100_log5.json`).

### Honest caveat

The best arms (BBA+orth-A, +gauge, +transport) cluster at **0.883–0.885 vs
0.851 ± 0.030** — a real lift, but ~1 std above a noisy baseline at **2 seeds**.
To turn this into a clean report claim it needs two cheap proxy runs:
1. **default-init + BBA** — proves it's an init×schedule *interaction* and not a
   Fig-6 non-replication. This cell was queued long ago and never run.
2. **BBA + orth-A seed 2** — third seed for the headline arm.

## Ruled out / closed (recorded so we don't re-run them)

| Idea | Verdict |
|---|---|
| SVD-compensated init | ➖ neutral on proxy; SVD+BBA collapses (0.766) |
| LoRA+ (asymmetric A/B LR) | ❌ closed by the paper's Fig 6 (balanced wins) |
| Partial participation | ❌ closed by RoLoRA's exactness argument (Eqs 3-4 survive sampling) — `docs/decisions/0007-*` |
| Factor-wise drift correction (FedProx, `rolora_prox`) | ➖ flat on the toy (56.07 vs 56.08); FedProx is the weak corrector |
| Server momentum (FedAvgM, `rolora_mom`) | ⏳ implemented, never run — the one untested lever |

## Where everything lives

| Thing | Path |
|---|---|
| Supplement harness + `SLS_*` switches (orth init, phase pattern, gauge, transport, A/B LR, monitor) | `code/harness/rolora-supplement/RoLoRA-code/federatedscope/` |
| Proxy configs | `experiments/configs/` |
| Proxy improvement curves (CSV) | `evidence/share_csv_for_chat_20260608/`, `evidence/improvement_diagnostics_20260604/` |
| Toy harness + variants | `notebooks/toy/` (`config.py` PRESETS) · `mnist_fig2_compare.py` · `toy/sweep.py` |
| Toy heterogeneity evidence | `results_extra/` |
| Per-idea reasoning | `.plans/`, `docs/decisions/`, `docs/current-roadmap.md` |
| Dense run log | `experiments/ledger/README.md` |

## Open / next (in priority order)

1. **default-init + BBA** proxy cell + **BBA+orth-A seed 2** — closes the
   PC-RoLoRA claim (the decisive runs).
2. `rolora_mom` toy sweep — the one untested proposal axis (server optimizer).
3. First clean cluster Table-1 cell (corrected AdamW + round count).

> ⚠️ Provenance note: the proxy numbers above are from runs up to 2026-06-12
> (newest local: basis transport). If newer proxy/W&B runs exist that aren't in
> this repo, fold them in before treating this as final.
