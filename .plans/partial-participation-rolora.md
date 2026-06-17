# RoLoRA under partial client participation

This ExecPlan is a living document. `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` must be kept up to date.

## Purpose / Big Picture

The paper's entire experimental section (§5) opens with *"Considering all
clients will participate in each round"* and never relaxes it. Table 1
(client-count robustness), Table 4 (local-step robustness), and Table 2
(Dirichlet non-IID) are **all full-participation**. Partial participation —
the default reality of cross-device FL — is the one robustness axis RoLoRA
claims a general framework for but never tests. That is our opening.

### Mechanism correction (verified in the harness, 2026-06-15)

The first pitch was "a returning client holds a *stale basis*, so FedAvg of its
update reintroduces the Eq.1≠2 interference." **That is wrong for this
harness.** Clients are download-then-train: on every round they are sampled,
`callback_funcs_for_model_para` overwrites local params with the broadcast
global model (`client.py:326-334`, `strict=share_local_model`). The basis is
re-synced each time a client participates, so there is **no per-client stale
basis** to transport. I am recording this so we do not relaunch the wrong
experiment.

What the investigation *did* find are two real, distinct effects under
`sample_client_rate < 1.0`:

1. **Phase desynchronization (artifact — must be neutralized, not measured).**
   The A/B phase is computed from a *local* counter `step_count`
   (`trainer.py:217`) that only advances when the client trains
   (`trainer.py:274`). Under sampling, two clients in the *same global round*
   hold different `step_count` and therefore freeze **different factors** — the
   server then averages a mix of A-updates and B-updates, manufacturing exactly
   the interference RoLoRA exists to remove. This is a harness-faithfulness bug,
   not a property of the paper's method (the paper defines phase by the *global*
   communication round, odd/even). The client already knows the global round
   (`self.state = round`, `client.py:335`); it just isn't used for phase.

2. **Cohort-mismatch incoherence (the real, RoLoRA-specific finding).** Even
   after we fix phase to track the global round, consecutive A- and B-rounds are
   fit by *different sampled cohorts*. An A-round moves global `A` (fit by cohort
   `S_t`) while global `B` is left expressed against the *old* `A`; the next
   cohort `S_{t+1}` downloads `(A_new, B_old)` and must re-fit `B` from an
   incoherent product `B_old @ A_new`. Under heterogeneity `S_t` and `S_{t+1}`
   differ in distribution, so the incoherence is real and should **grow as
   participation drops and as Dirichlet α shrinks**. Plain LoRA co-updates both
   factors on one cohort each round and has no cross-round cohort mismatch —
   so this is a liability specific to *alternation*, i.e. specific to RoLoRA.

### The fix (reuses code we already built)

Server-side function-preserving transport of the **global** B across each
A-update — `B_old → B_old @ A_old @ pinv(A_new)` — keeps the global product
coherent for the next cohort. This is exactly `apply_basis_transport`
(`federatedscope/core/sls_lora_transport.py`, env-gated `SLS_LORA_TRANSPORT=ls`).
Under full participation it tested ≈neutral (the same cohort co-adapts A and B
across the pair, washing the incoherence out). The hypothesis is that under
*partial* participation it finally has a job to do, because the cohorts differ.

> **Claim under test:** RoLoRA degrades under partial participation because
> alternation splits A- and B-updates across mismatched cohorts; server-side
> basis transport of the global B recovers the degradation, and the recovery
> grows with the stressor (lower participation, lower α).

## Arms (run at each participation × heterogeneity cell)

- **A0 — naive RoLoRA** (current `step_count` phase). The "what you get if you
  deploy RoLoRA as-shipped with sampling" baseline. Expected to break worst.
- **A1 — faithful RoLoRA** (global-round phase fix). The honest method
  baseline; isolates whether the *method* degrades, free of the phase bug.
- **A2 — faithful RoLoRA + transport** (`SLS_LORA_TRANSPORT=ls`). Our fix.

**The improvement comparison is A2 vs A1, never A2 vs A0.** Beating A0 would be
claiming credit for fixing our own harness bug. A0 is reported only to quantify
the naive-deployment failure. This is the confound discipline the SVD/transport
campaigns lacked.

## Falsifiable predictions

- **P1 (degradation exists):** A1 final accuracy drops monotonically as
  participation `p` falls (1.0 → 0.5 → 0.3 → 0.1) at fixed α, and the drop is
  larger at smaller α. If A1 does **not** degrade, there is nothing to fix.
- **P2 (transport recovers):** A2 > A1, and `(A2 − A1)` **grows** as `p` falls
  and α shrinks. A stressor-dependent gap is the signature that rules out the
  LR-retuning confound (a constant recipe effect cannot scale with `p`).
- **P3 (phase bug is real but not our credit):** A1 > A0 under sampling, and
  the monitor shows A0's sampled clients disagreeing on phase within a round
  while A1's agree. Confirms the artifact and that we neutralized it.

## De-risk probe FIRST (before any grid)

One cheap cell decides whether the campaign is alive:

```
# A1 faithful RoLoRA, p=0.3, Dirichlet alpha=0.5, seed 0, with monitor
SLS_PHASE_ROUND_SOURCE=global SLS_DEVICE=mps SLS_MONITOR=1 MODE=rolora \
  TAG=pp_a1_faithful_p0.3_a0.5_seed0 SEED=0 \
  bash scripts/run_supplement_arm.sh <pp config, sample_client_rate=0.3, lda alpha=0.5> seed 0
```

Pass conditions to continue: (i) monitor confirms all sampled clients agree on
phase each round (faithfulness fix works); (ii) A1 at p=0.3 is materially below
full-participation RoLoRA on the same split. **If A1 does not degrade, KILL the
campaign here** and ledger "RoLoRA is robust to partial participation in this
regime" — itself a clean, reportable negative.

## Kill criteria

- De-risk shows no A1 degradation at the strongest stressor → stop, ledger
  negative, do not run the grid.
- Full grid shows A2 ≈ A1 everywhere → transport is participation-neutral too;
  ledger as the second negative for transport and drop it from the method.
- A2 > A1 but the gap is **flat in p** (not growing) → suspect LR confound;
  re-tune LR per-arm at one cell before any claim. Do not headline until the
  gap is shown stressor-dependent.
- Any headline cell requires ≥3 seeds.

## Novelty check (carry over + extend from transport plan)

RoLoRA ablates schedules/LRs but never touches B when A changes and never
relaxes full participation. LoRA-A2 (2410.22815) dismisses product-space
reconciliation as "computationally unstable." ADF-LoRA (2511.18291) aligns the
frozen block under decentralized gossip — not a centralized partial-
participation problem. None address basis coherence across sampled cohorts.
Re-confirm against any newer FedLoRA partial-participation work before drafting.

## Implementation checklist

1. **Faithfulness fix:** thread the global round into the trainer and gate phase
   on it via `SLS_PHASE_ROUND_SOURCE=global` (default stays `step_count` so
   every prior result and baseline is byte-for-byte unchanged). Source the round
   from `self.state` (already set, `client.py:335`).
2. **Test:** unit test proving that with the global-round source, two trainers at
   different `step_count` but the same global round resolve to the **same**
   phase; and that the default (step_count) path is unchanged.
3. **Configs:** `proxy_qnli_..._pp.yaml` variants adding
   `federate.sample_client_rate` and `data.splitter: lda` with
   `splitter_args` α. Splitter exists: `core/splitters/generic/lda_splitter.py`.
4. **Smoke:** 2-client / 2-round smoke with monitor; confirm phase agreement and
   transport markers fire.
5. **De-risk probe** (above). Only on pass, build the p × α grid queue.

## Progress

- 2026-06-15: Plan created. Harness semantics verified (download-then-train;
  step_count phase; lda splitter present; sample_client_rate works in
  standalone). Mechanism corrected from "stale basis" to "cohort-mismatch
  incoherence + phase desync artifact."
- 2026-06-15: Faithfulness fix implemented and unit-tested.
  - `sls_phase_schedule.py`: added `phase_round_source_is_global` +
    `resolve_phase_round` (default == step_count, opt-in `global`).
  - `trainer.py`: phase + monitor round now use `resolve_phase_round`;
    `_sls_global_round` attr added.
  - `client.py`: sets `trainer._sls_global_round = self.state` before train.
  - `tests/test_sls_phase_round_source.py`: 8 tests green, incl. the P3
    cohort-agreement property; full SLS suite (35 tests) green.
  - Configs: `proxy_qnli_roberta_base_c50_r4_lr1e-2_pp.yaml` (p=0.3, lda α=0.5)
    and `smoke_supplement_pp.yaml`.
  - De-risk queue: `scripts/queues/run_pp_derisk_queue.sh` (3 gating cells).
  - Verified LDA train split IS label-aware (the `base_translator:144` warning
    is only about val/test prior alignment, which we don't depend on).
- 2026-06-15: **Splitter hang found and fixed (smoke caught it).** The generic
  `lda` splitter skews over `x['categories']`, which in `llm_dataset.py:125` is
  `pd.Categorical(df["category"]).codes` — the per-paragraph code, ~105k unique
  on QNLI. `dirichlet_distribution_noniid_slice` then loops `for k in
  range(105k)` and effectively hangs (observed 5h+ at 100% CPU, frozen right
  after the split, on BOTH cpu and mps attempts). Also semantically wrong:
  that is paragraph skew, not answer-label skew.
  - Fix: `federatedscope/contrib/splitter/lda_label_splitter.py` registers
    `lda_label`, which skews over the scalar QNLI answer label (`x['labels']`
    in {0,1}) and remaps unique labels to contiguous 0..C-1 so it can never
    hang. `categories` is consumed ONLY by splitters (grep-verified), so this
    is zero-risk to defaults/baselines. Both pp configs now use `lda_label`.
  - Validated in isolation (20s internal alarm): terminates instantly with
    unique-per-item categories; α=0.1 gives per-client class-1 fractions
    0.0..1.0 (real skew); α=100 collapses to near-IID. Regression test:
    `tests/test_lda_label_splitter.py`.
- 2026-06-15: **Faithfulness fix validated end-to-end** on the MPS pp smoke.
  Round 5 sampled cohort had step counts {4,3,2} (divergent participation)
  yet all resolved to `train A` under `SLS_PHASE_ROUND_SOURCE=global` — exactly
  the desync the fix targets. Clean exit, all 6 rounds, no errors.
- 2026-06-15: **De-risk probe launched** (`run_pp_derisk_queue.sh`, PID logged
  to `results/pp_derisk_queue_20260615_172025.log`). Cell 1 (A1 p=1.0) running.
  ~3-4h/cell on MPS; gating pair (cells 1-2) first. Read on completion:
  does A1 degrade at p=0.3 vs p=1.0? (P1). If not -> kill + ledger robustness.

## Surprises & Discoveries

- Download-then-train (`client.py:326-334`) invalidates the original
  stale-basis framing. The real RoLoRA-specific effect is cohort mismatch
  across the A/B alternation, not per-client staleness.
- The generic `lda` splitter is unusable as-is for these LLM datasets: it
  skews over the paragraph code (~105k pseudo-classes), which both hangs the
  dirichlet slicer for hours and is not the heterogeneity we want. Any future
  non-IID LLM run must use `lda_label` (answer-label skew), not `lda`.

## Decision Log

- 2026-06-15: A2-vs-A1 (not A2-vs-A0) is the only valid improvement comparison,
  to avoid claiming credit for fixing the phase-desync harness bug.
- 2026-06-15: Stressor-dependent gap (growing as p↓, α↓) adopted as the
  required signature to distinguish a real effect from LR re-tuning.

## Outcomes & Retrospective

(to be filled after the de-risk probe and, if it passes, the grid)
