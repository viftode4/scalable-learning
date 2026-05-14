# ADR 0003 — MNIST sanity-check training stability

**Status:** Accepted (2026-05-14)

## Context

The first full-MNIST run of `notebooks/mnist_fig2.py` (200 rounds, 5 clients, rank 1, lr 0.05, 20 local steps) crashed with `AssertionError: client A drifted from server` at round ~65 of the RoLoRA section. The deep-research plan warned this kind of break is exactly how RoLoRA silently degrades to broken-LoRA.

Root-cause investigation showed:
1. The exactness invariant (`torch.equal(client.frozen, server.frozen)`) was correct.
2. Around round 56–64 of RoLoRA, training loss diverged to NaN. `torch.equal` on NaN-poisoned tensors returns False (NaN != NaN by IEEE 754), so the assert fired AFTER the real problem (NaN training).
3. The unstable hyperparameters were: SGD with `lr=0.05`, no gradient clipping, rank-1 LoRA on a 2-layer MLP, full MNIST with 20 local steps per round per client. Gradient magnitudes explode after ~60 rounds.

## Decision

- Lower the default learning rate from `0.05` → `0.02`.
- Add L2 gradient-norm clipping (default `1.0`) in `local_train`.
- Make `assert_factor_identical` detect NaN explicitly and raise `RuntimeError` with a clearer message, so divergence surfaces at the source rather than as a confusing bit-equality failure.

## Consequences

- The `lr` and `grad_clip` knobs are exposed in `template_mnist_toy.yaml` and the CLI.
- 200-round full-MNIST runs now complete and reproduce the qualitative pattern from Figure 2:
  - LoRA (with the aggregation bug): plateaus around 46% accuracy.
  - FFA-LoRA: plateaus lower (~37%) — the frozen-A bottleneck.
  - RoLoRA: highest accuracy (~48%) and still climbing at round 200.
- The aggregation-invariant test (`tests/test_aggregation_invariants.py`) still passes and continues to enforce bit-equality for non-NaN paths.

## Alternatives rejected

- **Use Adam instead of SGD.** Hides the rank-1 fragility under adaptive scaling; SGD with grad-clip is more honest and closer to what the paper appears to use for the MNIST toy.
- **Raise the assert as-is.** Rejected: the user's first encounter with the script should not be a misleading drift error when the real cause is training divergence.
- **Skip the rank-1 setup.** Rejected: rank-1 is the paper's actual Fig. 2 setting and the sharpest sanity check for the alternation mechanism.
