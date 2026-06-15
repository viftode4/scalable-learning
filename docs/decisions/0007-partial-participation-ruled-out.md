# ADR 0007 — Partial participation is not an improvement angle for RoLoRA

**Status:** Accepted (2026-06-15). Ruled out on the paper's own theory before
spending cluster compute; the supporting harness work is kept as reusable
heterogeneity infrastructure.

## Context

Partial client participation (only a fraction of clients sampled per round) is
the default reality of cross-device federated learning, and the RoLoRA paper
never tests it — §5 opens with *"Considering all clients will participate in
each round."* That looked like an open gap worth an improvement: maybe RoLoRA
degrades under sampling and we could fix it (e.g. with the basis-transport code
we already had).

We investigated the supplement harness to scope an experiment and found:

1. **The harness is download-then-train.** A sampled client overwrites its local
   adapters with the freshly broadcast global model each round
   (`client.py` `callback_funcs_for_model_para`, `strict=share_local_model`).
   There is no per-client *stale basis* — the original framing was wrong.
2. **A real but separate harness artifact:** the A/B phase was derived from each
   trainer's local `step_count`, which desynchronizes under sampling. Fixed by
   `SLS_PHASE_ROUND_SOURCE=global` (ties phase to the global round). This is a
   harness faithfulness fix, not a property of the paper's method.

## Decision

**Do not pursue partial participation as an improvement.** It is closed by
RoLoRA's own exactness argument, not merely untested:

- Eqs 3-4 require only that the server broadcast the current frozen factor to
  whoever trains that round. Under client sampling with download-then-train, the
  sampled cohort all freeze the *same* just-broadcast factor, so **aggregation
  stays exact**. RoLoRA's core advantage survives partial participation
  unchanged; fewer clients only means fewer samples per round (slower
  convergence, the standard `m = Ω(q)` effect), not a broken mechanism.
- Plain LoRA loses the same samples *and* still pays the cross-factor
  interference RoLoRA removes. So RoLoRA degrades **less** than LoRA under
  sampling — this is a setting where RoLoRA looks good, not fragile. There is no
  improvement gap to close, only (at best) a reproduction-strengthening result.

## Consequences

- No partial-participation runs are queued. The full mechanism, the
  download-then-train finding, and the bugs found while scoping it (the
  5h `lda` splitter hang, the degenerate per-client eval metric) are recorded in
  `.plans/partial-participation-rolora.md`.
- The infrastructure built while scoping it is **kept** because it is reusable
  for the heterogeneity work that *is* live (per the scoreboard): the global-round
  phase source, the `lda_label` splitter (label-skew without the hang), and
  `make_global_eval` (evaluate on the full test set, not degenerate per-client
  shards). Configs `experiments/configs/*_pp.yaml` are kept as labeled
  scaffolding.
- General lesson, logged for the report: before treating an "untested setting"
  as an improvement opportunity, check whether the paper's existing theory
  already determines the outcome.
