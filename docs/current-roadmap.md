# Current RoLoRA project roadmap — 2026-06-03

This is the single current command document for the CS 4725 RoLoRA project.
It merges the paper targets, submitted-proposal commitments, current `main`,
Daniel's `origin/fix-rolora` branch, and the W&B audit from 2026-06-03.

## 0. Executive decision

We should **start improvement work now**, but we must be precise about what the
current baseline is.

- We **do have** a usable fast proxy baseline: QNLI / 50 clients / rank 4 /
  **RoBERTa-base** / 20 completed rounds from Daniel's W&B runs.
- We **do not yet have** paper-scale Table-1 reproduction: QNLI / 50 clients /
  rank 4 / **RoBERTa-Large** / 30 rounds / complete LoRA, FFA-LoRA, RoLoRA
  seed set.
- Therefore the next project phase is two-lane:
  1. **Proxy improvement lane** on RoBERTa-base/C50/r4/20 rounds, using the
     current W&B RoLoRA baseline.
  2. **Paper-scale reproduction lane** on RoBERTa-Large, run only for the
     minimum cells needed to support the final report.

Do **not** wait for perfect Table 1 reproduction before trying improvements.
Do **not** claim the current W&B runs are Table-1-scale RoBERTa-Large results.

## 1. Ground truths from the paper

Paper Table 1, QNLI / rank 4 / 50 clients / RoBERTa-Large:

| Method | Paper QNLI C50 r4 |
|---|---:|
| LoRA | 78.13 ± 5.13 |
| FFA-LoRA | 85.05 ± 0.34 |
| RoLoRA | 90.00 ± 0.63 |

Paper Section 5 says LR was tuned over `{5e-4, 1e-3, 2e-3, 5e-3, 1e-2,
2e-2, 5e-2, 1e-1}` and best-of-sweep was reported. Appendix Table 6 does not
pin optimizer or LR. This means our report can legitimately include a
reproducibility-audit story: the supplement alone is under-specified.

## 2. What the repo currently has on `main`

Current `main` is the safer base for real work because it has:

- corrected trainer alternation bookkeeping;
- classifier head kept trainable;
- W&B server/client logging instrumentation;
- DelftBlue policy encoded in docs and Slurm;
- paper-scale QNLI configs:
  - `experiments/configs/repro_qnli_c3_r4.yaml`
  - `experiments/configs/repro_qnli_c20_r4.yaml`
  - `experiments/configs/repro_qnli_c50_r4.yaml`
  - `experiments/configs/repro_qnli_c50_r8.yaml`
- C20 RoBERTa-Large sbatch templates under `slurm/`.

Current dirty/unpushed state is mostly presentation work plus the new W&B audit.
Do not do a destructive branch merge in this dirty working tree.

## 3. What Daniel's branch adds

Remote branch: `origin/fix-rolora`.

Useful pieces:

1. **RoBERTa-base C50 LR-sweep config/slurm pattern**
   - W&B run names match the visible W&B runs.
   - Important correction: branch C50 configs use
     `model.type: 'roberta-base@huggingface_llm'`, despite names like
     `repro_qnli_c50_r4_lr1e-2.yaml`.

2. **Toy/diagnostic improvement code**
   - `notebooks/toy/*`
   - `tests/test_toy_components.py`
   - `results_extra/*`
   - This is useful for the improvement narrative and pre-cluster sanity checks.

3. **Improvement idea catalog**
   - `docs/deep-research-improvements.md`
   - Useful, but must be filtered to the submitted proposal directions.

Do **not** merge `origin/fix-rolora` wholesale. It is behind current `main` and
would delete/overwrite useful current docs and weaken current trainer/client
instrumentation.

Files to avoid taking from Daniel's branch without manual review:

- `code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/trainer/trainer.py`
- `code/harness/rolora-supplement/RoLoRA-code/federatedscope/core/workers/client.py`
- deletions under `docs/presentations/deepspeed-ulysses/`
- deletion of `docs/decisions/0006-supplement-reproducibility-gap.md`
- deletion of local runner scripts such as `scripts/run_supplement_arm.sh`

## 4. W&B evidence now available

Evidence bundle:

```text
evidence/wandb_qnli_c50_r4_20260603/
```

W&B project:

```text
scalable-learning-7/sls-rolora-repro
```

Visible run group:

```text
qnli_c50_r4
```

Interpretation after branch inspection:

- These are most likely **RoBERTa-base**, not RoBERTa-Large, because Daniel's
  matching branch configs use `roberta-base@huggingface_llm`.
- Runs contain histories for rounds `0..19`, so they completed their configured
  20-round trajectories.
- W&B state `crashed` is consistent with Slurm walltime termination at/after
  finalization under the 4h `gpu-a100-small` cap.

Current usable proxy baseline:

| Method/LR | Seeds | Result |
|---|---:|---:|
| RoLoRA `lr=1e-2` | 0,1,2 | mean final/best ≈ 0.851; best 0.8713 |
| RoLoRA `lr=5e-3` | 0,1,2 | mean best ≈ 0.851 |
| LoRA best visible | mixed | best ≈ 0.571 |
| FFA-LoRA best visible | mixed | best ≈ 0.570 |

Use this as the **proxy control baseline** for improvement experiments.
Do not use it as Table 1 reproduction evidence.

## 5. Submitted proposal commitments

The project promised three improvement directions, all preserving RoLoRA's
alternating structure:

1. **Improved initialization** — orthogonal / SVD-based init for A.
2. **Separate learning rates for A and B** — LoRA+-style asymmetric LRs.
3. **Adaptive server-side optimization** — lightweight federated optimizer
   instead of plain averaging.

The first improvement to implement should be **orthogonal-A initialization**:

- cheapest;
- does not require data leakage decisions;
- works with current PEFT version by direct patching;
- directly supports the proposal.

## 6. Critical implementation fact: PEFT version

The supplement venv currently reports:

```text
peft 0.3.0
```

This version does **not** support PiSSA/OLoRA config switches. Therefore:

- do not assume `init_lora_weights: pissa` works;
- do not make PiSSA the first implementation;
- first patch orthogonal-A directly in the harness/adapter initialization path;
- PiSSA/SVD can be a later follow-up after either upgrading PEFT or writing a
  careful custom init/residual implementation.

## 7. Immediate integration tasks

Use a clean worktree/branch, not the dirty current tree:

```bash
git worktree add ../scalable-learning-integrate main
cd ../scalable-learning-integrate
git checkout -b integrate-daniel-useful-work
```

Then selectively import only safe Daniel artifacts:

```bash
git checkout origin/fix-rolora -- \
  docs/deep-research-improvements.md \
  notebooks/toy \
  tests/test_toy_components.py \
  results_extra
```

Manual port, not blind checkout:

- create clearly named RoBERTa-base proxy configs, e.g.
  `experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml`;
- create matching proxy sbatch templates under `slurm/`, or a generator script;
- preserve current `main` trainer/client files;
- preserve current `main` docs and ADR 0006.

Verification for integration:

```bash
uv run pytest tests/test_toy_components.py tests/test_mnist_fig2.py
uv run ruff check tests/test_toy_components.py notebooks/toy
```

Acceptance criteria:

- toy tests pass;
- no current `main` harness fixes are reverted;
- proxy configs clearly say `RoBERTa-base`;
- W&B metadata records model/config/git/optimizer/LR/rank/seed/job id.

## 8. Immediate experiment sequence

### Step A — lock the proxy baseline in the ledger

Record the current W&B runs as:

```text
Proxy baseline PB1:
QNLI / RoBERTa-base / 50 clients / rank 4 / 20 rounds
vanilla RoLoRA, lr=1e-2, seeds 0,1,2
mean final ≈ 0.851, best 0.8713
```

Add this to:

- `experiments/ledger/README.md`
- `docs/progress.md` claim ledger
- report notes if needed

### Step B — improve W&B metadata before new runs

Future runs must log:

- model type (`roberta-base` vs `roberta-large`);
- config path;
- git SHA;
- optimizer type;
- LR;
- rank;
- LoRA alpha/dropout;
- seed;
- Slurm job ID;
- total rounds;
- eval frequency;
- alternation mode;
- init variant;
- server optimizer variant.

### Step C — implement orthogonal-A init

Add an opt-in config/env switch such as:

```text
SLS_LORA_INIT=orthogonal_a
```

Behavior:

- default remains vanilla LoRA/RoLoRA initialization;
- with `orthogonal_a`, every LoRA-A matrix is orthogonally initialized;
- LoRA-B remains zero for real PEFT LoRA so the pretrained base model is
  preserved at step 0;
- W&B logs `init_variant=orthogonal_a`.

Verification:

- add/extend a test that checks LoRA-A Gram structure on a tiny PEFT model or
  isolated helper;
- run a tiny smoke with `SLS_LORA_INIT=orthogonal_a` and confirm no model-load
  regression.

### Step D — run proxy improvement seed 0

Run:

```text
QNLI / RoBERTa-base / C50 / r4 / 20 rounds / RoLoRA / lr=1e-2 / seed 0
```

Compare against current vanilla seed 0:

```text
rolora_lr1e-2_seed0 final = 0.8214
```

Decision gate:

- If orthogonal-A seed 0 is clearly worse: stop or try one more LR.
- If it is within noise but better early: run seeds 1,2 for variance/curve story.
- If it improves final by >= 1 point or materially improves early convergence:
  fan out seeds 0,1,2 and make it the first improvement result.

### Step E — run A/B LR proxy only after orthogonal-A

Try LoRA+ ratios on RoBERTa-base proxy:

```text
lr_B / lr_A ∈ {2, 4, 8, 16}
```

Start seed 0 only. Replicate only the best ratio.

### Step F — adaptive server optimizer last

Only after init/LR baselines are understood, implement active-factor server
optimizer:

- separate state for A rounds and B rounds;
- start with server momentum or FedAdam-style update;
- log moment policy (`persist-same-factor` vs `reset-on-phase-switch`).

## 9. Paper-scale minimum after proxy signal

Once one improvement has signal on RoBERTa-base proxy, spend paper-scale compute
on the smallest RoBERTa-Large set that can support the final report.

Minimum paper-scale target:

```text
QNLI / RoBERTa-Large / C50 / r4 / 30 rounds
```

Run at least:

1. vanilla RoLoRA, seeds 0,1,2;
2. best improvement, seeds 0,1,2.

If compute allows, add:

3. LoRA baseline, seeds 0,1,2;
4. FFA-LoRA baseline, seeds 0,1,2.

Because 4h `gpu-a100-small` is tight, RoBERTa-Large C50/r4/30 should probably
use full `gpu-a100` or reduced eval frequency. The 4h small partition is good
for proxy/control curves, not guaranteed for full paper-scale completion.

## 10. Report narrative we are building

Final report should not be an experiment dump. The narrative should be:

1. **Reproduction audit:** The paper/supplement under-specifies optimizer/LR;
   naive shipped settings fail; tuned settings are required.
2. **Baseline reproduction/proxy:** RoLoRA learns robustly under high-client
   C50/r4 proxy; LoRA/FFA are weak under the current proxy setup.
3. **Improvement:** Better phase initialization and/or phase-specific optimizer
   choices can improve early convergence/variance while preserving RoLoRA's
   alternating communication advantage.
4. **Limitations:** Full Table 1 reproduction is compute-bound; FlexLoRA is not
   reproduced because it is absent from the supplement; RoBERTa-base proxy
   results are not RoBERTa-Large paper numbers.

## 11. Next 10 concrete actions

1. Create clean integration worktree.
2. Selectively import Daniel's toy code/tests/results and improvement notes.
3. Port Daniel's C50 LR sweep as explicitly named RoBERTa-base proxy configs.
4. Add current W&B proxy baseline to the experiment ledger.
5. Patch future W&B metadata to remove model/config ambiguity.
6. Implement `SLS_LORA_INIT=orthogonal_a` on current `main` harness.
7. Add an orthogonal-A init test/smoke.
8. Run proxy orthogonal-A seed 0 at RoLoRA `lr=1e-2`.
9. If promising, run proxy orthogonal-A seeds 0,1,2 and plot against PB1.
10. Prepare RoBERTa-Large C50/r4 vanilla vs best-improvement jobs for DelftBlue.

## 12. Stop rules

- Stop trying to merge Daniel's branch wholesale; only selective import is safe.
- Stop treating current W&B C50 runs as RoBERTa-Large unless Slurm logs prove it.
- Stop expanding LR grids if seed-0 proxy does not show signal.
- Stop spending full-GPU time before metadata logging is fixed.
- Stop claiming an improvement unless it has a matched vanilla RoLoRA baseline,
  same model/dataset/clients/rank/rounds/seed policy, and a curve.

## 13. June 2 team-chat context

The WhatsApp/Teams/Canva conversation from 2026-06-02 is mostly about the
DeepSpeed/Ulysses paper-presentation deck, not the RoLoRA experiment plan.

Relevant interpretation:

- Daniel's comments about “momentan mi se pare ca nu prea zicem nimic”,
  “sequence wise”, ZeRO, sparse/dense attention, sections 4.4–4.6, and results
  slides are about the Ulysses presentation content.
- The Canva link and Claude share are presentation artifacts. They explain why
  there are local/unpushed edits under `docs/presentations/deepspeed-ulysses/`.
- This chat does **not** add new RoLoRA experiment evidence and should not be
  conflated with Daniel's `origin/fix-rolora` branch.
- For RoLoRA, Daniel's relevant technical branch remains `origin/fix-rolora`;
  the useful pieces are still the RoBERTa-base C50 proxy sweep pattern,
  toy/improvement code, and improvement notes.

Presentation cleanup action:

- If we want the repo to mirror the submitted June 2 deck, compare the final
  submitted Canva/PPTX against the local `docs/presentations/deepspeed-ulysses/`
  files and commit only the final artifact/source notes. This is separate from
  the RoLoRA project execution lane.

## 14. May 28 RoLoRA/team-intent context

The May 28 chat is directly relevant to the RoLoRA project direction.

Relevant content:

- Daniel wants thesis/work in distributed ML / federated learning and sees this
  project as useful preparation.
- The team intent is to “bagam tare” on Scalable Learning if there are strong
  ideas.
- Daniel explicitly wanted code/results reuse, real confidence intervals, and a
  stronger presentation with plots.
- His main technical interest: **super-heterogeneous data** — the regime where
  each client sees only a narrow label slice, e.g. one-label-per-client toy
  setting. This is exactly where LoRA/FFA-LoRA collapse and RoLoRA survives.
- Daniel's early empirical observation: initialization seemed to give roughly a
  **3% accuracy bonus** on a single run, and he planned multi-seed reruns to see
  whether it generalizes.

Implications for the project:

1. The improvement story should be framed around **high heterogeneity / client
   specialization**, not generic hyperparameter tuning.
2. The toy lane is not just a side artifact; it is the diagnostic microscope for
   the heterogeneity claim.
3. Confidence intervals / multi-seed plots are mandatory for the final story,
   especially for initialization gains.
4. The first improvement to scale from toy/proxy to RoBERTa should remain
   **initialization**, because it already showed early signal and matches the
   proposal.
5. A good final thesis is:

   > RoLoRA's alternating structure is especially valuable under extreme client
   > heterogeneity; smarter initialization of the shared basis A can further
   > improve early convergence/variance while preserving the communication
   > structure.

Updated priority order after this chat:

1. Integrate Daniel's toy/plot/CI code.
2. Reproduce/verify the multi-seed toy initialization result.
3. Run the same initialization variant on the RoBERTa-base C50 proxy baseline.
4. Only then spend RoBERTa-Large compute.
5. Keep A/B LR and server optimizer as secondary improvement axes unless init
   fails to generalize.

## 15. Go/no-go for improvements based on existing evidence

Decision: **GO for improvements now**, with the scope limited to proxy/toy first.

Passed evidence already available:

1. **Extreme-heterogeneity toy baseline passed**
   - Source: `origin/fix-rolora:results_extra/baselines_n5_r100_log5.json`
   - Config: 10 clients, 1 label/client, 100 rounds, rank 16, 5 seeds.
   - Mean final accuracies:
     - LoRA: 0.6230 ± 0.0212 CI95
     - FFA-LoRA: 0.6251 ± 0.0146 CI95
     - RoLoRA: 0.8389 ± 0.0179 CI95
     - Centralized ceiling: 0.9749 ± 0.0015 CI95
   - Interpretation: the heterogeneity regime Daniel described is already
     reproduced: standard baselines collapse relative to RoLoRA.

2. **Orthogonal-A toy improvement passed as a multi-seed signal**
   - Source: `origin/fix-rolora:results_extra/orth_a_n5_r100_log5.json`
   - Same config as above.
   - RoLoRA + orthogonal A mean final accuracy: 0.8719 ± 0.0114 CI95.
   - Mean improvement over base RoLoRA: +0.0330 absolute accuracy.
   - Paired seed deltas: +0.0410, +0.0416, +0.0703, -0.0065, +0.0187.
   - Interpretation: Daniel's “~3% accuracy bonus from initialization” is real
     in the existing toy evidence, not just a single-run hunch.

3. **RoBERTa-base C50 proxy RoLoRA baseline exists**
   - Source: `evidence/wandb_qnli_c50_r4_20260603/`.
   - Likely from Daniel's `origin/fix-rolora` RoBERTa-base configs.
   - Config: QNLI, RoBERTa-base, 50 clients, rank 4, 20 completed rounds.
   - RoLoRA `lr=1e-2`, seeds 0/1/2: mean ≈ 0.851, best 0.8713.
   - Interpretation: there is enough proxy baseline to test whether the toy
     initialization gain transfers to language-model fine-tuning.

Not passed yet:

- RoBERTa-Large Table-1 reproduction is **not** passed.
- LoRA/FFA-LoRA RoBERTa-base proxy baselines are weak and should be treated as
  negative/diagnostic evidence, not paper-comparable baselines.
- PiSSA/SVD init is **not** implementation-ready because the supplement venv is
  on PEFT 0.3.0 and lacks PiSSA/OLoRA switches.

Immediate consequence:

- We can start the improvement phase now, but the first real experiment should
  be the direct transfer test:

  ```text
  Orthogonal-A RoLoRA vs vanilla RoLoRA
  QNLI / RoBERTa-base / 50 clients / rank 4 / 20 rounds / lr=1e-2
  seed 0 first, then seeds 0/1/2 if promising
  ```

- The target to beat for seed 0 is vanilla `rolora_lr1e-2_seed0 = 0.8214`.
- The target to beat for 3-seed mean is vanilla `rolora_lr1e-2 mean ≈ 0.851`.

## 16. June-11 draft artifact bundle generated on 2026-06-03

The June-11 draft target is now explicitly compute-constrained: do **not** block
writing on full RoBERTa-Large Table 1 reproduction. The repo now has concrete
figures/tables for the draft:

### W&B RoBERTa-base proxy bundle

Command:

```bash
uv run python scripts/plot_wandb_proxy.py
```

Evidence:

```text
evidence/wandb_qnli_c50_r4_20260603/figures/
```

Generated report artifacts:

- `proxy_a_server_accuracy_convergence.{png,pdf}`
- `proxy_b_final_best_by_method_lr.{png,pdf}`
- `proxy_c_rolora_lr_sweep.{png,pdf}`
- `proxy_d_rolora_per_seed_trajectories.{png,pdf}`
- `proxy_e_walltime_crash_audit.{png,pdf}`
- `proxy_plot_summary.md`

Caption constraint: these are QNLI / RoBERTa-base / 50 clients / rank 4 / 20
round proxy results, **not** RoBERTa-Large Table-1 reproduction.

### Toy heterogeneity + orthogonal-A bundle

Command:

```bash
uv run python scripts/plot_toy_heterogeneity.py
```

Evidence:

```text
evidence/toy_heterogeneity_20260603/figures/
```

Generated report artifacts:

- `toy_g_heterogeneous_baselines_curves.{png,pdf}`
- `toy_g_heterogeneous_baselines_final_acc.{png,pdf}`
- `toy_h_orthogonal_a_paired_delta.{png,pdf}`
- `toy_plot_summary.md`

Key numbers: LoRA 62.30 ± 2.12%, FFA-LoRA 62.51 ± 1.46%, RoLoRA 83.89 ±
1.79%, centralized 97.49 ± 0.15%; orthogonal-A RoLoRA 87.19 ± 1.14%, a +3.30
pp final-accuracy gain over vanilla RoLoRA in this toy setting.

### Updated draft control docs

- `docs/progress.md` now has supported-proxy claim C6 and supported-toy claim C7.
- `experiments/ledger/README.md` records the generated evidence bundles.
- `report/README.md` maps the June-11 draft figure/table slots to concrete paths.

Orthogonal-A implementation status: `SLS_LORA_INIT=orthogonal_a` now patches
PEFT LoRA initialization directly in the supplement adapter builder; the default
baseline remains unchanged, and `tests/test_sls_orthogonal_lora_init.py` verifies
LoRA-A orthogonality plus zero LoRA-B without loading a HuggingFace model.

Next draft-critical action: run the smallest orthogonal-A GLUE proxy transfer
against the RoLoRA `lr=1e-2` control using
`experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml`; only then spend
time on one paper-scale RoBERTa-Large attempt.


## 17. Improvement execution update — 2026-06-03 evening

Correction to the execution stance: do **not** freeze improvements until June 9.
For the June-11 draft, improvement work is the priority after consolidating the
existing results.

Current state:

- Existing results are handled in `report/draft_results_brief_20260603.md`.
- Orthogonal-A has supported toy evidence: +3.30 pp final accuracy over vanilla
  RoLoRA on the 10-client one-label/client MNIST setting.
- Orthogonal-A is implemented in the supplement harness via
  `SLS_LORA_INIT=orthogonal_a`.
- Separate A/B learning rates are implemented via `SLS_LORA_LR_A` and
  `SLS_LORA_LR_B`; default behavior is unchanged when unset.
- Adaptive server optimization is command-ready through FederatedScope FedOpt in
  `experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2_fedopt_adam.yaml`.
- Real supplement smoke proof passed in
  `results/overnight_smoke_improve_orth_ab.log`: orthogonal-A initialized LoRA
  matrices and A/B LR groups switched correctly across B/A rounds.
- Draft-critical orthogonal-A proxy seed 0 is running at
  `results/overnight_proxy_orth_a_c50_r4_lr1e-2_seed0.log` with W&B group
  `qnli_c50_r4_improvements`.

Run order remains:

1. Monitor orthogonal-A proxy seed 0 and compare to vanilla seed-0 control
   (`0.8214` final/best).
2. If competitive, run orthogonal-A seeds 1/2 and make the matched improvement
   plot.
3. If clearly worse, run the A/B LR seed-0 probe.
4. Use FedOpt seed 0 as the third-axis fallback/ablation, not before the first
   two axes are interpretable.

## 9. Current improvement thesis — factor-aware phase control

Updated 2026-06-04 after the first BBA + orthogonal-A proxy result.

The strongest current direction is **factor-aware phase control for RoLoRA**:
keep RoLoRA's alternating low-rank structure, but replace the paper's blind
round-by-round A/B alternation with a schedule or controller that gives the
right factor more update bandwidth at the right time.

Why this is a good project direction:

- **Scales like the paper:** fixed schedules such as `BBA` add no trainable
  parameters, no extra communication payload, and no extra forward/backward
  passes. They should transfer from RoBERTa-base proxy runs to RoBERTa-Large
  paper-scale runs better than methods that require extra model state.
- **Works at small scale too:** the same mechanism is testable on smoke runs,
  MNIST/toy heterogeneity, and RoBERTa-base C50 proxy runs before spending
  DelftBlue time.
- **Explains the method, not just the number:** internal monitors can show
  `ΔA=0` on B rounds, `ΔB=0` on A rounds, update norms, aggregation drift, and
  convergence phases. That gives us a mechanistic report story.
- **Fits the submitted proposal:** it composes cleanly with orthogonal/SVD init,
  separate A/B learning rates, and phase-aware server optimization.

Current evidence:

| Variant | Scope | Seed | Result | Status |
|---|---|---:|---:|---|
| Orthogonal-A + default AB | QNLI / RoBERTa-base / C50 / r4 / 20 rounds | 0 | test `0.829398`, val `0.823473` | complete |
| Orthogonal-A + `BBA` phase schedule | same | 0 | test `0.885411`, val `0.889313` | complete; strongest current run |
| Orthogonal-A + `BBA` phase schedule | same | 1 | pending | running as replication |

### 9.1 Test ladder

Do not launch multiple full MPS runs at once. Keep one full proxy run active,
then queue the next experiment from this ladder.

1. **Replicate the current winner**
   - Run `BBA + orthogonal-A` seeds 1 and 2.
   - Required before claiming a real improvement in the draft.

2. **Fixed phase-schedule sweep**
   - Compare cheap seed-0 variants:
     - `BBA` — current winner.
     - `BBBA` — even more B bandwidth; tests whether A is still too frequent.
     - `BBAA` — tests whether paired A refresh helps after B builds signal.
     - `SLS_B_WARMUP_ROUNDS=2` then AB — tests whether the win is just early
       B warmup or persistent B bias.
   - Keep only variants that beat default AB by round 8-10 or show a clear
     late-curve advantage.

3. **BBA + asymmetric LoRA learning rates**
   - First probe:
     ```bash
     SLS_LORA_INIT=orthogonal_a SLS_PHASE_PATTERN=BBA \
     SLS_LORA_LR_A=0.005 SLS_LORA_LR_B=0.01 SLS_DEVICE=mps SLS_MONITOR=1 \
     MODE=rolora TAG=proxy_bba_orth_ab_lr_A5e-3_B1e-2_seed0 SEED=0 \
       bash scripts/run_supplement_arm.sh \
       experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml seed 0
     ```
   - If BBA is already giving B more rounds, A may need a smaller LR and B may
     tolerate the base LR or a larger LR. This is the cleanest composition with
     proposal axis 2.

4. **Phase-aware FedOpt / FactorFedOpt**
   - Standard FedOpt may carry momentum for factors that should be frozen in a
     given phase. The better variant is **phase-aware server optimization**:
     update only the active LoRA factor plus classifier on each server step;
     keep the inactive factor exactly synchronized.
   - This is the clean version of proposal axis 3 and should scale because it
     changes server optimizer dynamics, not model size.

5. **Adaptive Phase Controller (APC-RoLoRA)**
   - Replace fixed `BBA` with a deterministic controller using logged signals:
     - repeat B if B-update norm / aggregation drift remains high and eval is
       still improving;
     - force an A round at least every 3 rounds to refresh the subspace;
     - switch to A when B updates plateau or client drift spikes.
   - This is the more novel "layer over RoLoRA": not random alternation, but a
     feedback controller over the two-factor training dynamics.
   - Keep the first version simple and deterministic so it is reportable.

6. **Gradient/SVD-informed A refresh**
   - After 1-2 B-heavy warmup rounds, estimate the dominant client-update
     subspace and refresh/projection-regularize A with an orthogonal basis.
   - More ambitious and more complex; use only after the fixed schedule and A/B
     LR tests are stable.

7. **Heterogeneity-aware aggregation normalization**
   - Log client update norm/cosine spread by phase.
   - If a few clients dominate B updates, test update clipping or norm-normalized
     averaging for active LoRA factors only.
   - This connects directly to the paper's heterogeneous-data motivation.

### 9.2 Metrics needed for the draft

Already logging:

- server/client train/test/val accuracy and loss;
- fairness std/min/max;
- phase markers;
- local update norms for LoRA-A, LoRA-B, classifier;
- aggregation drift/update norms.

Add if time permits:

- singular values / effective rank of `B @ A` per round;
- cosine similarity / dispersion across client active-factor updates;
- ratio of classifier update norm to LoRA update norm;
- wall-clock per round and time-to-threshold accuracy.

### 9.3 Report claim shape

If seeds 1/2 confirm seed 0, the draft claim should be:

> RoLoRA's random/fixed AB alternation underuses the B factor early in training.
> A simple factor-aware phase schedule (`BBA`) combined with orthogonal-A
> initialization improves the RoBERTa-base QNLI C50/r4 proxy without increasing
> communication, parameter count, or per-round compute. Internal monitors verify
> exact phase freezing and show that the gain comes from controlled factor-update
> dynamics rather than an implementation artifact.

This is strong because it is **scalable**, **small-scale-testable**, and
**mechanistically explainable**.

## 10. Robust LoRA/RoLoRA stack — borrow ideas, do not copy blindly

The right framing is not "steal code/results". It is: **borrow proven ideas,
cite them, implement cleanly in our harness, and ablate each layer**. Most LoRA
variants are manipulating the same few degrees of freedom:

- how A/B are initialized;
- which factor/subspace is trained when;
- how large each update is;
- how client updates are aggregated;
- whether rank/subspace/magnitude is adapted over time.

Our opportunity is to build a robust federated LoRA recipe by composing the
lowest-risk pieces around RoLoRA's alternating structure.

### 10.1 Minimum viable banger for the June-11 draft

Do **not** try to implement every LoRA paper. For the draft, keep the stack:

```text
RoLoRA core
+ orthogonal-A initialization
+ product-preserving orthogonal gauge fix
+ factor-aware phase schedule/controller (BBA first)
+ optional A/B asymmetric LR ablation
+ optional phase-aware FedOpt ablation
```

This is enough to be a real contribution because it is:

- small and scalable;
- compatible with the paper's method;
- measurable on proxy and toy scales;
- explainable with our monitor logs;
- connected to the proposal commitments.

### 10.2 Idea sources to borrow from, one at a time

| Borrowed idea family | What to take | How it becomes our method | Risk |
|---|---|---|---|
| Orthogonal/OLoRA/PiSSA-style init | Better starting subspace for A | Already implemented as `SLS_LORA_INIT=orthogonal_a`; later add SVD/gradient-informed refresh | Low |
| LoRA reparameterization / gauge freedom | `BA` is invariant under `A = R A_orth`, `B <- B R` | `SLS_LORA_GAUGE=orthogonal_a` keeps A row-orthogonal after aggregation while preserving the effective adapter exactly | Low-medium |
| LoRA+ | Different LRs for A and B | `SLS_LORA_LR_A/B`, tested on top of BBA | Low |
| FedAdam/FedYogi/FedOpt | Server momentum/adaptivity | Make it phase-aware so inactive factor stays frozen/synchronized | Medium |
| DoRA / magnitude-direction split | Separate direction from magnitude | Possible later as a scalar/gain per LoRA module, but not before draft unless core results are done | Medium-high |
| AdaLoRA / dynamic rank | Allocate rank where useful | Use singular spectrum/effective-rank monitor first; implementation later | High |
| SCAFFOLD/FedProx-style FL stabilization | Reduce client drift under heterogeneity | Start with active-factor update norm clipping/normalization, not full control variates | Medium-high |
| Robust aggregation | Prevent outlier clients dominating | Phase-specific clipping or norm-normalized averaging on active LoRA factors | Medium |

### 10.3 Step-by-step build order

1. **Prove replication:** finish BBA+orthogonal-A seeds 1/2.
2. **Test non-redundant math layer:** BBA+orthogonal-A with
   `SLS_LORA_GAUGE=orthogonal_a`. This is not another B-prioritized schedule;
   it uses LoRA's product-preserving gauge freedom to stabilize the factorization
   after federated aggregation.
3. **Ablate schedule only if needed:** BBA vs BBBA vs BBAA vs
   B-warmup-then-AB. Note: the RoLoRA paper already reports aggressive
   B-prioritized/A-prioritized and unequal-LR variants, so this is secondary
   unless our proxy contradicts the paper.
4. **Add LoRA+ layer:** BBA+orthogonal-A with asymmetric A/B LR.
5. **Add server optimizer layer:** phase-aware FedAdam/FedYogi, active factor
   only.
6. **Add diagnostics before complexity:** singular spectrum, client-update cosine
   dispersion, active-factor norm spread.
7. **Only then try heavier ideas:** adaptive controller, SVD refresh, dynamic
   rank, DoRA-style magnitude.

Each step must answer one question. If it does not beat the previous layer or
explain a failure, we stop and move on.

### 10.4 What is too much right now

Too much before June 11:

- full dynamic rank allocation;
- full SCAFFOLD/control-variate FL;
- DoRA-style architecture rewrite;
- multiple GLUE tasks plus multiple client counts plus all improvements;
- paper-scale RoBERTa-Large sweeps for every idea.

Not too much:

- one strong proxy improvement with 3 seeds;
- one toy/heterogeneity confirmation;
- one or two clean ablations explaining why it works;
- one paper-scale RoBERTa-Large attempt for transfer plausibility.

### 10.5 Final method target

If the results hold, the final method can be framed as:

```text
Phase-Controlled RoLoRA (PC-RoLoRA)
= RoLoRA alternating low-rank adaptation
+ orthogonal/SVD-informed A initialization
+ factor-aware phase schedule/controller
+ optional phase-aware server optimizer
```

This is a better target than inventing a totally new LoRA variant because it
keeps the paper's core idea but fixes the weak point we can now observe: blind
alternation does not allocate update bandwidth according to the factor dynamics.

## 11. More novel option — Adaptive Phase-Controlled RoLoRA

Updated 2026-06-04. This is the more novel version of the current BBA result.
BBA is a strong fixed schedule; the research contribution can be stronger if we
use it as evidence for a **feedback controller** over RoLoRA's factor dynamics.

Working name:

```text
APC-RoLoRA = Adaptive Phase-Controlled RoLoRA
```

Core idea:

> RoLoRA should not alternate A/B blindly. It should choose which factor to train
> from observable low-rank update dynamics: active-factor update norm,
> aggregation drift, recent validation gain, and a no-starvation constraint.

This is more novel than just importing OLoRA/LoRA+/FedOpt because those improve
initialization, factor learning rates, or server optimization. APC-RoLoRA changes
**the temporal control policy** of alternating federated low-rank training.

### 11.1 Minimal controller v1

Keep it deterministic and cheap:

```text
Inputs per round:
- previous phase
- validation/test proxy improvement over last aggregate
- active LoRA factor update norm
- active LoRA factor client-drift norm
- current streak length for A or B

Rules:
1. Start with B for at least two rounds.
2. Prefer another B round if B updates are still large and validation is not
   degrading.
3. Force an A round after at most two consecutive B rounds.
4. After an A round, return to B unless B drift is exploding.
5. Never allow either factor to be skipped for more than three rounds.
```

This should reproduce BBA-like behavior early but can adapt later if the update
norms or validation curve say the fixed schedule is wrong.

### 11.2 Why this is plausibly novel but feasible

- **Novel axis:** schedule/control policy for alternating federated LoRA factors.
- **Low implementation cost:** current monitor already logs the signals needed.
- **No scaling penalty:** no extra trainable parameters, no communication
  increase, no inference-time change.
- **Small-to-large ladder:** test on RoBERTa-base proxy, then one RoBERTa-Large
  transfer run if it works.
- **Clean ablation:** AB vs BBA vs APC directly isolates the controller.

### 11.3 Required ablations for APC claim

Minimum acceptable set:

| Variant | Purpose |
|---|---|
| Orthogonal-A + AB | baseline with same init |
| Orthogonal-A + BBA | best fixed schedule found so far |
| Orthogonal-A + APC | novel adaptive schedule |
| Orthogonal-A + APC + A/B LR | optional stacked best variant |

If APC only matches BBA, it is still useful as a negative/neutral result: the
simple fixed schedule is enough. If APC beats BBA or has lower variance/time to
threshold, it becomes the final method.

### 11.4 What not to claim

Do not claim global publication-level novelty without a deeper literature audit.
The safe claim is:

> To our knowledge in this project scope, we introduce a factor-aware adaptive
> phase schedule for RoLoRA in federated fine-tuning and show it can improve or
> match the best fixed schedule without extra communication.
