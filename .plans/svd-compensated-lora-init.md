# Add compensated SVD/PiSSA initialization for RoLoRA adapters

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository does not currently have a checked-in PLANS.md file. This document follows the local create-plan skill requirements and is self-contained.

## Purpose / Big Picture

The project already has one safe initialization improvement, `SLS_LORA_INIT=orthogonal_a`, which changes LoRA-A while keeping LoRA-B zero so the adapter starts as an identity update. Daniel asked us to check two papers about SVD-based LoRA initialization and, specifically, whether a non-zero adapter product must be subtracted from the frozen base weights. After this plan is implemented, we will be able to run RoLoRA with a compensated SVD/PiSSA-style initialization that starts from the same effective pretrained model output while giving LoRA-A and LoRA-B principal-singular-vector values instead of Gaussian/zero values.

The observable behavior is: with `SLS_LORA_INIT=svd_compensated`, every supported PEFT LoRA linear layer has non-zero A and B factors, but `base_weight_after + lora_delta_after` reconstructs the original pretrained base weight at initialization. A smoke run should print an `[sls-rolora] SLS_LORA_INIT=svd_compensated` marker and finish two rounds. W&B remains covered by the separate fixed logging tests and by the real proxy run, because the smoke script intentionally disables W&B.

## Progress

- [x] (2026-06-08 20:23 CEST) Downloaded the two requested papers into `docs/research/`: `paper-frlora-iclr2025.pdf` and `paper-pissa-arxiv2404.02948.pdf`.
- [x] (2026-06-08 20:23 CEST) Read the relevant methods sections and confirmed that both papers compensate the base weight when the SVD-initialized adapter product is non-zero.
- [x] (2026-06-08 20:23 CEST) Inspected current adapter code in `code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/model/adapter_builder.py`; only `default` and `orthogonal_a` are implemented today.
- [x] (2026-06-08 20:46 CEST) Added red-first tests in `tests/test_sls_svd_lora_init.py` for compensated SVD math, aliases, unsupported fan-in/out, double-apply guard, and PEFT 0.3 inline-weight compatibility.
- [x] (2026-06-08 20:46 CEST) Implemented `SLS_LORA_INIT=svd_compensated` in `adapter_builder.py`, preserving existing `default` and `orthogonal_a` behavior.
- [x] (2026-06-08 20:47 CEST) Ran targeted tests, full pytest, py_compile, diff check, new-test ruff, and supplement smoke; all required checks passed. Adapter-file ruff still reports unrelated pre-existing style issues.
- [x] (2026-06-13 audit) Actual QNLI/RoBERTa-base/C50/r4 proxy runs already exist for `SLS_LORA_INIT=svd_compensated` at lr `1e-2`, seeds 0/1/2. They average **85.70 ± 0.59%** final server test accuracy. This is only a small/neutral gain over vanilla RoLoRA and below the stronger orthogonal-A+BBA / adaptive-refresh actual-model results, so SVD should not be the headline unless a lower-LR follow-up changes the picture.

## Surprises & Discoveries

- Observation: Daniel's concern is correct. SVD/PiSSA initializes both LoRA factors non-zero, so the adapter product is non-zero and would change the pretrained model's initial output unless the base weight is compensated.
  Evidence: FRLoRA Algorithm 1 computes SVD of the pretrained weight, initializes global B and A from the principal singular components, then sets the working base weight to `W_hat_0 = W_0 - B_0 A_0`. PiSSA decomposes `W = W_res + AB` and freezes `W_res`, so the initial forward still equals the original pretrained `W`.

- Observation: PEFT 0.10.0, the version pinned in this repo, does not expose PiSSA as a built-in `LoraConfig.init_lora_weights` option. Its local signature accepts `True`, `False`, `gaussian`, or `loftq` only.
  Evidence: `uv run python -c 'from peft import LoraConfig; import inspect; print(inspect.signature(LoraConfig))'` shows `init_lora_weights: bool | Literal['gaussian', 'loftq'] = True`.

- Observation: PEFT applies LoRA as `base_layer(x) + lora_B(lora_A(x)) * scaling`, where `scaling` is usually `lora_alpha / r`. The compensation must therefore subtract the actual scaled PEFT delta, not an unscaled mathematical product.
  Evidence: local PEFT source at `.venv/lib/python3.11/site-packages/peft/tuners/lora/layer.py` computes `output_tensor = transpose(weight_B @ weight_A, fan_in_fan_out) * self.scaling[adapter]` and adds `lora_B(lora_A(...)) * scaling` in forward.

- Observation: The supplement venv uses PEFT 0.3.0, whose LoRA linear layer stores the frozen base weight directly as `module.weight` rather than under `base_layer` / `get_base_layer`.
  Evidence: first smoke failed with `SLS_LORA_INIT=svd_compensated found no PEFT LoRA linear layers`; inspecting `code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/lib/python3.9/site-packages/peft/tuners/lora.py` showed `class Linear(nn.Linear, LoraLayer)`. Added a regression test for this inline-weight shape and then smoke initialized 24 layers.

## Decision Log

- Decision: Implement this as a proposal-safe initialization variant, not as full FRLoRA residual accumulation.
  Rationale: The submitted proposal commits to improved initialization while preserving RoLoRA's alternating structure. Full FRLoRA would update frozen base weights every communication round and reset adapters each round, which is a larger algorithm change and a confound for the current reproduction/improvement story.
  Date/Author: 2026-06-08 / Codex

- Decision: Use canonical environment value `SLS_LORA_INIT=svd_compensated`, with aliases `pissa` and `pissa_compensated` accepted for convenience.
  Rationale: `svd_compensated` describes exactly what our implementation does without overclaiming the full PiSSA or FRLoRA algorithm. The aliases make Daniel's paper terminology easy to use in commands.
  Date/Author: 2026-06-08 / Codex

- Decision: Preserve initial effective weights by making the PEFT-scaled delta equal the top-r SVD reconstruction and subtracting that same scaled delta from the frozen base weight.
  Rationale: The papers preserve the initial pretrained function by representing `W` as a frozen residual plus a trainable principal low-rank part. In PEFT, the forward includes a scale factor, so exact preservation must be checked using PEFT's real `get_delta_weight`/forward convention.
  Date/Author: 2026-06-08 / Codex

- Decision: Compute SVD in float32 on CPU initially.
  Rationale: RoBERTa query/value matrices are small enough for exact SVD in the proxy path, and CPU SVD avoids polluting the small 10 GB A100 MIG CUDA allocator. If this becomes too slow for paper-scale RoBERTa-Large, add an optional randomized/low-rank SVD variant later.
  Date/Author: 2026-06-08 / Codex

## Outcomes & Retrospective

Implemented and smoke-verified on 2026-06-08. The code now supports `SLS_LORA_INIT=svd_compensated` for both root PEFT 0.10-style layers and the supplement PEFT 0.3 inline-weight layers. The 20-round QNLI/RoBERTa-base proxy was later run for seeds 0/1/2 at lr `1e-2`; see `docs/improvement-handoff-2026-06-13.md` for the current interpretation.

## Context and Orientation

This repository reproduces and improves RoLoRA, a federated LoRA method. LoRA adds a low-rank trainable update to a frozen pretrained matrix. For a linear layer with pretrained weight `W`, PEFT LoRA computes an effective weight `W + scaling * (B @ A)`, where `A` has shape `rank x input_dim`, `B` has shape `output_dim x rank`, and `scaling = lora_alpha / rank` in the standard setting.

The current harness wraps HuggingFace models with PEFT in `code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/model/adapter_builder.py`. The function `_apply_sls_lora_init(model, init_variant)` runs after `get_peft_model(...)` creates LoRA parameters. Today it supports `default` as a no-op and `orthogonal_a`, which orthogonalizes LoRA-A and zeroes LoRA-B. Current tests for this live in `tests/test_sls_orthogonal_lora_init.py`. The toy MNIST FFA-LoRA convention in `tests/test_init_conventions.py` asserts B-zero initialization for that toy path; the new SVD-compensated variant will intentionally have non-zero B and should be tested separately rather than weakening the toy convention.

The two downloaded papers are stored locally as:

- `docs/research/paper-frlora-iclr2025.pdf`, from OpenReview id `e0rQRMUhs7`, title `Federated Residual Low-Rank Adaptation of Large Language Models`.
- `docs/research/paper-pissa-arxiv2404.02948.pdf`, from arXiv id `2404.02948`, title `PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models`.

The main paper takeaway for implementation is: if `A` and `B` are initialized from singular vectors, then `B @ A` is not zero. To avoid changing the model before training starts, the frozen base weight must be changed from `W` to a residual `W_res = W - effective_lora_delta`. FRLoRA does this in its Algorithm 1 before federated training starts. PiSSA does the same conceptually by decomposing the original matrix into frozen residual components plus trainable principal components.

This plan deliberately does not implement full FRLoRA. Full FRLoRA accumulates residual low-rank updates into base weights every round and reinitializes local LoRA factors, which changes the federated update rule. Here we only add the compensated SVD/PiSSA initialization as an improvement axis compatible with the existing RoLoRA alternation code.

## Why This Could Improve RoLoRA

The baseline LoRA initialization is safe but inefficient. It sets LoRA-B to zero and LoRA-A to random noise, so the adapter update is initially zero and the model starts exactly at the pretrained model. This also means the first gradient for LoRA-A is zero when LoRA-B is zero, and the first LoRA-B updates are expressed through a random LoRA-A basis. In ordinary centralized LoRA this can waste early optimization steps. In RoLoRA the effect is more important because the algorithm intentionally trains only one LoRA factor per round. A B-round with random A means clients all update B in a random low-rank input subspace; an A-round early in training depends on whatever B happened to become after client-local updates and federated averaging.

SVD/PiSSA-style initialization changes the basis from random directions to principal directions of the pretrained weight. For a pretrained matrix `W = U diag(s) Vh`, the top singular vectors are the directions where that layer already has the strongest action. Initializing `B` and `A` from `U[:, :r]`, `sqrt(s[:r])`, and `Vh[:r, :]` means the trainable low-rank factors begin aligned with important pretrained directions rather than arbitrary noise. In RoLoRA terms, the first B-round trains B against a meaningful A basis, and if the phase schedule ever starts with or reaches an A-round, A also has a non-zero B partner so its gradients are not initially dead.

The compensation is what makes this a fair improvement instead of a hidden model perturbation. If both factors are non-zero, then `scaling * B @ A` is non-zero. Without compensation, the initial effective model is `W + scaling * B @ A`, which is not the original pretrained RoBERTa and could improve or harm results for the wrong reason. With compensation, the frozen base weight becomes `W_res = W - scaling * B @ A`, so the initial effective model is exactly `W_res + scaling * B @ A = W`. The only thing that changes at step zero is the parameterization of the same function: the principal component is now trainable and the residual component is frozen.

This is attractive for the proposal because it is a single-axis initialization improvement. It preserves the existing RoLoRA alternation and federation code. It does not introduce FRLoRA's per-round base-weight residual accumulation, which would be a different algorithm and would make it harder to attribute any gain to initialization.

## How It Might Fail

This is not guaranteed to win. The top singular directions of a pretrained weight are important for the pretrained model, but the best downstream fine-tuning update can live partly in lower singular directions. If the rank is only 4, moving exactly those top components into the trainable adapter may be too restrictive for QNLI. Federated factor averaging is also not the same as averaging dense weight updates, so non-zero A and B may interact with RoLoRA's gauge and aggregation behavior differently from zero-B LoRA. The compensated base weight also means adapter-only export is no longer a plain LoRA adapter on the original model unless we convert it after training; for in-process evaluation this is fine, but for sharing checkpoints the trained adapter should be represented as the difference from the initial principal adapter.

Because of these risks, this plan treats SVD-compensated initialization as an experiment with kill criteria, not as an assumed final improvement. If it does not beat vanilla RoLoRA or orthogonal-A on the proxy ladder, we should keep the result as a negative ablation and either try a safer A-only SVD basis variant or move effort to the other proposal axes.

## Plan of Work

First, add focused unit tests before touching implementation. Create `tests/test_sls_svd_lora_init.py`. The test should build a tiny dummy PEFT-like model with one LoRA-wrapped linear layer. The dummy layer must expose enough of the real PEFT interface to exercise our code: a base layer with a two-dimensional `weight`, `lora_A` and `lora_B` module dictionaries containing `torch.nn.Linear` modules without bias, an `active_adapters` list containing `default`, a `scaling` dictionary, and a `get_delta_weight(adapter)` method that returns `scaling[adapter] * (B.weight @ A.weight)`. Use a base weight with known shape, for example `6 x 5`, rank `2`, and scaling `4.0`. Before initialization, clone the original base weight. After `_apply_sls_lora_init(model, 'svd_compensated')`, assert that A and B are both non-zero, `base_after + delta_after` is close to the original base weight, and `delta_after` is close to the top-r reconstruction from `torch.linalg.svd(original_weight.float(), full_matrices=False)`.

Second, extend `_normalise_sls_lora_init` in `adapter_builder.py`. It should return `svd_compensated` for input values `svd`, `svd_compensated`, `pissa`, and `pissa_compensated`. Existing accepted values for `default` and `orthogonal_a` must remain unchanged. Unknown values should still raise `ValueError`.

Third, implement helper functions in `adapter_builder.py` near `_apply_sls_lora_init`. Keep these helper names stable so tests can target them if needed:

- `_iter_lora_linear_layers(model)` should yield modules that look like PEFT LoRA linear layers by checking for `lora_A`, `lora_B`, and a callable `get_base_layer` or a `base_layer` attribute.
- `_get_lora_base_layer(module)` should return the underlying frozen base linear layer.
- `_apply_svd_compensated_lora_init(model)` should iterate over each LoRA layer and active adapter, compute the rank from `module.r[adapter]` or from `lora_A[adapter].weight.shape[0]`, run SVD on `base_layer.weight.detach().float().cpu()`, set LoRA-B and LoRA-A from the principal singular vectors, and subtract the actual PEFT delta from the base weight.

The factorization should be scale-aware. Let the base weight be `W` with shape `out_dim x in_dim`, exact SVD `W = U diag(s) Vh`, target rank `r`, and PEFT scale `c = module.scaling[adapter]`. The desired effective LoRA delta is the principal reconstruction `W_pri = U[:, :r] diag(s[:r]) Vh[:r, :]`. Because PEFT multiplies `B @ A` by `c`, initialize factors so `c * (B @ A) = W_pri`. One simple construction is:

    sqrt_s_over_c = sqrt(s[:r] / c)
    B = U[:, :r] * sqrt_s_over_c[None, :]
    A = sqrt_s_over_c[:, None] * Vh[:r, :]

After assigning these weights, compute the actual delta through the same convention PEFT uses, preferably `module.get_delta_weight(adapter)` when available, and subtract it from `base_layer.weight.data`. Then `base_layer.weight.data + actual_delta` should reconstruct the cloned original `W` within tolerance.

Fourth, keep logging minimal but useful. At the end of the SVD init, print one marker line like:

    [sls-rolora] SLS_LORA_INIT=svd_compensated: initialized 24 LoRA layers; max reconstruction error 1.23e-06; max delta norm ...

The smoke script already greps for `[sls-rolora]`, so this marker makes the new variant visible in logs and preserves the existing operational flow.

Fifth, do not change training phase logic yet. In `code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/trainer/trainer.py` the alternation modes already decide whether LoRA-A or LoRA-B is trainable each round. With the new initialization both factors start non-zero, but the B/A phase switching can remain unchanged. The first B-round will update B while A remains the shared principal-singular-vector factor.

Sixth, after implementation, add a ledger entry or short note only if we actually run an experiment. Do not claim an improvement from initialization until at least one matched 20-round proxy run exists with fixed W&B logging.

## Concrete Steps

Work from repository root `/Users/vliftode/personal/scalable-learning`.

1. Add tests:

    uv run pytest tests/test_sls_orthogonal_lora_init.py -q

This should pass before any implementation changes. Then add `tests/test_sls_svd_lora_init.py` with the dummy layer tests described above. Run it and expect failure because `svd_compensated` is not implemented yet:

    uv run pytest tests/test_sls_svd_lora_init.py -q

2. Implement the new variant in:

    code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/model/adapter_builder.py

Keep changes local to this file unless tests show a real need elsewhere.

3. Run unit and regression tests:

    uv run pytest tests/test_sls_svd_lora_init.py tests/test_sls_orthogonal_lora_init.py tests/test_init_conventions.py -q
    uv run pytest tests/test_sls_lora_gauge.py tests/test_supplement_runner.py tests/test_wandb_round_logging.py -q
    uv run python -m py_compile code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/model/adapter_builder.py

Expected result: all selected tests pass, and py_compile prints no output.

4. Run a local CPU smoke, assuming the supplement venv and cached RoBERTa-base artifacts are already present:

    SLS_LORA_INIT=svd_compensated \
    SLS_MONITOR=1 \
    MODE=rolora \
    LOG_PREFIX=smoke_svd_compensated \
    bash scripts/smoke_supplement.sh rolora

Expected result: `results/smoke_svd_compensated_rolora.log` contains the new `[sls-rolora] SLS_LORA_INIT=svd_compensated` marker and two federated rounds complete. The smoke script intentionally sets `WANDB_MODE=disabled`, so W&B step warnings are checked by `tests/test_wandb_round_logging.py` and the first real proxy run instead.

5. If smoke passes and the team agrees, run one matched proxy experiment with fixed W&B logging:

    SLS_LORA_INIT=svd_compensated \
    SLS_MONITOR=1 \
    MODE=rolora \
    TAG=proxy_svd_compensated_c50_r4_lr1e-2_seed0 \
    SEED=0 \
    WANDB_MODE=online \
    bash scripts/run_supplement_arm.sh \
      experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml seed 0

Use the matching slurm path instead if this needs to run on DelftBlue, but keep the same environment variables and tag. Do not spend RoBERTa-Large cluster time on this until the proxy run has complete server rounds 0 through 19 in local CSV and W&B.

6. Evaluate the proxy run against controls. Parse local logs/CSVs, not W&B alone, and produce a small table with final validation accuracy, final test accuracy, best validation accuracy, best test accuracy at the best-validation round, and early convergence such as area under the validation curve or first round reaching a fixed validation threshold. Compare in this order: vanilla RoLoRA with the same config and seed, `SLS_LORA_INIT=orthogonal_a`, and then `SLS_LORA_INIT=svd_compensated`. If the single-seed SVD run is clearly worse than both vanilla and orthogonal-A, do not spend more compute. If it is close but has faster early convergence, run seed 1 before deciding. If it is better by roughly one percentage point or more on validation or materially improves early convergence without hurting final test accuracy, run seeds 1 and 2 before making any report claim.

## Validation and Acceptance

The implementation is acceptable when all of the following are true:

- `tests/test_sls_svd_lora_init.py` proves that compensated SVD initialization leaves the effective initial weight unchanged: `base_after + delta_after` equals the original base weight within floating-point tolerance.
- Existing `orthogonal_a` tests still pass, proving we did not break the current improvement path.
- Existing FFA-LoRA toy initialization convention tests still pass, proving the new non-zero B behavior is isolated to the explicit SVD variant.
- The smoke run produces an `[sls-rolora] SLS_LORA_INIT=svd_compensated` marker and completes.
- W&B logging remains fixed: future runs log round-indexed metrics under explicit `server/round` and `client_NN/round` axes, not by passing global `step=round`.

A proxy experiment is acceptable for analysis only when its local CSV and W&B both show server rounds 0 through 19. If W&B is missing intermediate rounds but local CSV is complete, the experiment may be used for internal debugging but should not be used as clean evidence for final plots without saying so.

The improvement is strong enough to keep only if it satisfies at least one of these conditions on the proxy ladder: final validation/test is better than vanilla RoLoRA by about one percentage point or more; or time-to-threshold/early area-under-curve is clearly better while final accuracy remains within noise of vanilla; or it combines positively with the already promising BBA phase schedule without introducing logging or aggregation anomalies. A single seed is exploratory only. A final-report claim needs either multiple seeds or very careful language that it is a proxy observation.

## Idempotence and Recovery

The SVD init runs only at model construction time and is deterministic for a fixed pretrained weight. Re-running a smoke or proxy from scratch is safe. Do not call `_apply_sls_lora_init(..., 'svd_compensated')` twice on the same already-compensated model object, because the second call would decompose the residual base rather than the original pretrained base and would subtract a second delta. Tests should exercise a fresh dummy model per test case.

If a run fails because exact CPU SVD is too slow, keep the implementation but do not escalate to paper-scale. Add a follow-up option such as `SLS_LORA_SVD_METHOD=lowrank` using `torch.svd_lowrank` or a randomized SVD routine, then validate that the reconstruction-preservation test is adjusted to the approximate top-r delta. If a run fails because unsupported LoRA module types appear, make the helper raise a clear error naming the unsupported module rather than silently producing wrong weights.

If the compensation math looks wrong in a smoke, temporarily set `SLS_LORA_INIT=orthogonal_a` or unset `SLS_LORA_INIT` to return to the already verified path.

## Artifacts and Notes

Downloaded paper artifacts:

    docs/research/paper-frlora-iclr2025.pdf       3.9M  25 pages
    docs/research/paper-pissa-arxiv2404.02948.pdf 2.0M  34 pages

Local PEFT check:

    peft 0.10.0
    init_lora_weights: bool | Literal['gaussian', 'loftq'] = True

Relevant current implementation file:

    code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/model/adapter_builder.py

Current dirty working tree already contains the W&B logging fix files and exported evidence CSVs. Keep SVD implementation changes scoped; do not accidentally rewrite or remove those existing W&B changes.

## Interfaces and Dependencies

The new public interface is the environment/config value:

    SLS_LORA_INIT=svd_compensated

Accepted aliases should include:

    SLS_LORA_INIT=svd
    SLS_LORA_INIT=pissa
    SLS_LORA_INIT=pissa_compensated

The implementation should remain in `adapter_builder.py` and expose at least these internal helpers:

    def _normalise_sls_lora_init(value): ...
    def _apply_sls_lora_init(model, init_variant): ...
    def _apply_svd_compensated_lora_init(model): ...

`_apply_sls_lora_init` must return the same model object it receives, matching the existing `orthogonal_a` behavior.

This plan depends on PyTorch's `torch.linalg.svd`, PEFT 0.10.0's LoRA layer conventions, and the existing supplement smoke script. It must not require upgrading PEFT or adding a new dependency.

Revision note, 2026-06-08: Initial plan written after downloading and inspecting the FRLoRA and PiSSA papers. The main design choice is to implement only compensated SVD initialization now, leaving full FRLoRA residual accumulation as a separate possible future experiment.
