# Toy-first RoLoRA improvement bank — 2026-06-13

## Purpose

Use the MNIST Figure-2 toy as the fast filter before spending RoBERTa compute.
The paper's main language experiments assume full participation and random/IID
partitions for the client-count headline, but the mechanism we can iterate on
fastest is client drift / factor parameterization. This bank separates:

1. **normal participation runs**: cheap evidence for initialization and phase
   schedule candidates;
2. **stale-factor stress runs**: a deliberately harsher partial-participation
   model where sampled clients receive only the active factor, exposing the
   basis/coordinate mismatch that ordinary full downlink hides.

## Mathematical hypotheses

Let a layer's LoRA update be `Δ = A B` in the toy orientation.

- **SVD-compensated init** decomposes the frozen pretrained weight
  `W = W_res + A_0 B_0` with `A_0 B_0` equal to the rank-r principal SVD
  component. The initial function is unchanged, because the base is compensated
  to `W_res = W - A_0 B_0`, but the trainable low-rank subspace starts aligned
  with high-energy directions of `W` instead of a random Kaiming basis. This also
  removes the zero-gradient-at-A-at-step-zero pathology of `B=0`.

- **Orthogonal-A init** keeps `B=0` and preserves the original random-A scale
  while conditioning the basis (`AᵀA` diagonal). It is lower-risk than SVD
  because it keeps the initial adapter exactly zero and does not modify the
  frozen base weight.

- **B-prioritized schedules under improved init** test whether Figure 6's
  negative result for B-priority is specific to default random-A initialization.
  In the toy bank, BBA/BBBA/BBBBA did not beat SVD and only roughly matched
  orthogonal-A, so the current fast evidence does not promote them above init.

- **Factor-wise drift correction (new, 2026-06-15).** Reasoning the paper
  misses: RoLoRA's alternation makes *aggregation* exact (it removes the
  cross-factor interference `avg(AB) ≠ avg(A)avg(B)`), but on every round it
  still runs vanilla FedAvg on the *trained* factor over heterogeneous clients.
  So **within-factor client drift is completely untouched** — the standard FL
  problem — and that is the residual gap to the centralized ceiling under skew
  (toy: RoLoRA ≈84 vs ceiling ≈97). Our own orth-A win is consistent with this:
  a better-conditioned frozen basis means the trained factor drifts less.
  The fix: add a FedProx-style proximal anchor `(mu/2)||w − w_t||²` to the
  single alternating factor each round (`w_t` = the just-broadcast global
  value). Because only one low-rank factor trains per round, this is cheap and
  adds **zero communication**. Variants: `rolora_fedprox_lo/…/_hi` (mu sweep)
  and `orth_fedprox` (stacks the best init with drift correction). FedProx is
  the *weak* drift corrector; if it shows any signal, the stronger SCAFFOLD-style
  control-variate version is the next step. Kill if no mu beats vanilla RoLoRA
  by ≥1 pp over the bank's matched seeds.

- **Basis transport** solves the coordinate mismatch after a basis update:
  `B_new = argmin_B ||A_new B - A_old B_old||_F`. It is only expected to help
  when clients or server state can hold coefficients in a stale basis. Under
  ordinary sampled-client full downlink, the current harness re-syncs both
  factors for sampled clients, so the stale-basis failure is intentionally tested
  with `sync_policy=active_only`.

## Evidence

Commands run:

```bash
uv run python scripts/run_toy_improvement_bank.py \
  --split label_shard \
  --variants lora,ffa,rolora,orth,orth_bba,orth_bbba,orth_bbbba,orth_transport,partial,partial_stale,partial_stale_transport,svd,svd_bba \
  --seeds 0,1,2 --rounds 60 --local-steps 10 --clients 10 --rank 16 \
  --subset 10000 --test-subset 2000 \
  --outdir evidence/toy_improvement_bank_20260613_002709_label_shard

uv run python scripts/run_toy_improvement_bank.py \
  --split label_shard \
  --variants rolora,orth,orth_bba,svd,partial_stale,partial_stale_transport \
  --seeds 3,4 --rounds 60 --local-steps 10 --clients 10 --rank 16 \
  --subset 10000 --test-subset 2000 \
  --outdir evidence/toy_improvement_bank_20260613_003519_label_shard_confirm
```

Combined table:

`evidence/toy_improvement_bank_label_shard_combined.md`

Key rows:

| Candidate | Evidence | Promotion decision |
|---|---:|---|
| SVD-compensated RoLoRA | `64.20 ± 1.99%` vs vanilla RoLoRA `56.08 ± 4.41%` on 5 matched seeds | **Promote first** |
| Orthogonal-A RoLoRA | `58.40 ± 1.44%` vs vanilla RoLoRA `56.08 ± 4.41%` on 5 matched seeds | **Promote as low-risk init baseline** |
| Orthogonal-A + BBA | `57.92 ± 1.16%` on 5 matched seeds | Keep as secondary interaction check, not primary |
| Stale-factor transport | `43.61 ± 6.08%` vs stale baseline `38.96 ± 6.83%` final; best `47.82 ± 4.02%` vs `44.21 ± 3.25%` | Mechanism stress test; not a normal partial-participation claim yet |

## Next proper runs

Important correction after re-checking the actual logs: SVD-compensated init
was already run on the QNLI/RoBERTa-base/C50/r4 proxy at lr `1e-2`
(`results/overnight_proxy_svd_compensated_c50_r4_lr1e-2_seed{0,1,2}.log`).
It reached `85.70 ± 0.59%` final server test accuracy, which is only a
small/neutral gain over vanilla RoLoRA and below the stronger actual-model
orthogonal-A+BBA and adaptive-refresh results. The toy result should therefore
not be used to reset priorities.

1. **Finish the strongest actual-model cells first.**
   Orthogonal-A+BBA and adaptive refresh+orthogonal-A are currently stronger
   than SVD on the actual QNLI proxy. See
   `docs/improvement-handoff-2026-06-13.md`.

2. **Orthogonal-A RoBERTa-base proxy, QNLI C50 r4, missing seeds.**
   This is the safer initialization story if SVD is unstable or too hard to
   export cleanly.

3. **SVD lower-LR sanity only if compute is spare.**
   A lr `5e-3` SVD run can close the initialization-ablation loop, but it is no
   longer the first run to launch.

4. **Partial/stale-basis transport only as a controlled stress experiment.**
   First verify in the real FederatedScope harness whether `sample_client_rate`
   with ordinary broadcast sends both factors to sampled clients. If yes, a
   stale-basis claim needs an explicit communication-saving active-only downlink
   variant; ordinary partial participation is not enough.

## Kill criteria

- Drop a candidate if it loses to vanilla RoLoRA by ≥1 pp on the matched proxy
  after two seeds.
- Drop B-prioritized schedules if default-init+BBA reproduces Figure 6's
  negative result and SVD/orthogonal-A alone captures the gain.
- Do not claim partial-participation improvement unless the run semantics
  actually leave a returning client with a stale frozen factor.
