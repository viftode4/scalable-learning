# Function-preserving basis transport for RoLoRA A-rounds

This ExecPlan is a living document. `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` must be kept up to date.

## Purpose / Big Picture

Phase-resolved diagnostics show RoLoRA's A-rounds erasing B-round accuracy
gains (e.g. round 2 B: +5.5 pp, round 3 A: −5.7 pp in the orth-A seed-0
extract). Mechanism: the adapter is `Δ = B @ A`; a B-round moves the function
inside the span of A, but an A-round replaces the basis underneath B's frozen
coefficients, so the learned function is corrupted by `B @ (A_new − A_old)`.

The fix implemented here transports the coefficients into the new basis after
every A-round aggregation:

    B' = argmin ||B' A_new − B A_old||_F = B @ A_old @ pinv(A_new)

The transport matrix `M = A_old @ pinv(A_new)` is r×r (4×4 in the proxy cell),
deterministic given the old/new global factors, adds **zero communication**,
zero parameters, and is env-gated (`SLS_LORA_TRANSPORT=ls`,
optional `SLS_LORA_TRANSPORT_BETA` in [0,1], default 1.0) so baselines stay
unchanged. It runs in the server aggregator **before** the orthogonal gauge,
so transport restores the function and gauge then re-conditions the
factorization without touching the product.

Novelty check (2026-06-11, `docs/research/README.md`): RoLoRA's paper ablates
schedules/LRs but never touches B when A changes; LoRA-A2 (arXiv 2410.22815)
explicitly dismisses product-space reconciliation as "computationally
unstable" (our closed-form r×r solve avoids the decomposition they feared);
ADF-LoRA (arXiv 2511.18291) aligns the frozen block under DFL gossip — a
problem that does not exist in centralized RoLoRA. Neither transports
coefficients across a basis update.

## Falsifiable predictions (the experiment design)

- **P1** vanilla-AB + transport: the per-A-round accuracy drops vanish;
  final accuracy ≥ vanilla-AB control (0.851 ± 0.030).
- **P2** BBA + orth-A + transport vs BBA + orth-A (0.885/0.881): if BBA's
  advantage came from avoiding A-round damage, transport should make balanced
  AB competitive with BBA (run P1 vs BBA controls), and stacking transport on
  BBA should change little. P1 ≈ BBA-level would *explain* paper Figure 6.
- **P3** svd_compensated at lr=1e-2 + transport: the SVD seed-1-style
  collapse (~0.66) was basis thrash, so transport should stabilize it.

Queue: `results/run_transport_queue.sh` (P1/P2/P3, seed 0 each, W&B group
`qnli_c50_r4_transport`). Evaluate exactly like prior arms: local
`*__server_metrics.csv` rounds 0–19 as primary evidence, per-A-round deltas
from phase markers as the mechanism plot.

## Progress

- [x] (2026-06-11) Novelty check vs LoRA-A2 + ADF-LoRA; PDFs added to
  `docs/research/` and README table updated.
- [x] (2026-06-11) Red-first tests `tests/test_sls_lora_transport.py`
  (10 tests: exact preservation when the new basis spans the old,
  pinv-reference match, B-round no-op, beta damping, rank-deficient fallback,
  env gating/aliases, aggregator integration for partial A-round payloads,
  composition with `SLS_LORA_GAUGE=orthogonal_a`).
- [x] (2026-06-11) Implemented
  `federatedscope/core/sls_lora_transport.py` and hooked it into
  `ClientsAvgAggregator.aggregate` before the gauge block, with
  `[sls-transport]` marker + monitor/W&B emission.
- [x] (2026-06-11) 49 tests green (transport + gauge + init + phase + LR +
  runner), py_compile clean, ruff clean on new files.
- [ ] Supplement smoke with `SLS_LORA_TRANSPORT=ls`
  (`results/smoke_transport_rolora.log`, expect `[sls-transport]` marker on
  the A-round).
- [ ] Run `results/run_transport_queue.sh` (P1 → P2 → P3) and ledger results.

## Surprises & Discoveries

- Observation: `torch.linalg.lstsq` (gelsy) returns the min-norm solution for
  a rank-deficient `A_new` instead of failing, which would silently zero B.
  Resolution: explicit `matrix_rank(A_new) < r` guard falls back to identity
  and counts `transport_fallback_count`.

## Decision Log

- Decision: transport runs server-side in the aggregator, before the gauge.
  Rationale: `self.model.state_dict()` is the pre-aggregation state, so
  `A_old` is free; transport restores the function, then gauge
  re-orthonormalizes A while preserving the (restored) product. Both write
  their A/B pairs back into the aggregate payload the server loads.
  Date/Author: 2026-06-11 / Claude
- Decision: kill criteria — if P1 shows no reduction in per-A-round damage
  *and* no accuracy gain over vanilla-AB at seed 0, stop after one seed and
  ledger it as a negative mechanism test. If P1 works but P3 still collapses,
  keep transport and drop the SVD-init revival.

## Outcomes & Retrospective

(to be filled after the queue completes)
