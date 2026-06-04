# Project progress board

This is the live control surface for the RoLoRA final project. Keep it short,
current, and evidence-backed. Every row needs a concrete next action and an
evidence path.

## Current thesis

We deliver a compute-constrained reproduction and proxy-scale improvement study
of RoLoRA: audit the released supplement, verify the patched harness locally,
show RoLoRA's robustness on heterogeneous toy/proxy settings, and test
proposal-compatible `A`/`B` interventions under the hardware budget we actually
have.

## Strategy lock

- **Primary path for the 2026-06-11 draft:** compute-constrained,
  diagnostic-first QNLI story with explicit proxy labels.
- **Paper-scale path:** one or a few RoBERTa-Large attempts only when they do
  not displace draft-critical evidence.
- **Primary dataset for this draft:** QNLI, because the supplement pipeline and
  W&B runs are already wired and locally verified there.
- **Stretch dataset:** MNLI only after the QNLI/proxy story is ledgered.
- **Forbidden pivots:** unrelated prior-project framing, a different paper-presentation paper,
  more account/fairshare churn before the confirmed DelftBlue share is tried, or Llama-2-7B as a baseline path.

## Fallback triggers

Switch from paper-scale reproduction-first to smaller diagnostic-first execution
when any trigger below fires:

| Trigger | Default response |
|---|---|
| RoBERTa-Large feasibility fails due to memory/runtime on DelftBlue | Record failure in the ledger; retry on `gpu-v100`; if still blocked, fall back to QNLI/RoBERTa-base or smaller RoBERTa-Large cells with full diagnostics. |
| 3-client RoBERTa-Large baseline misses the paper by more than ±2% without an explainable setup cause | Stop broad scaling; debug baseline comparability before more cluster spend. |
| 50-client degradation pattern is not reproduced by the agreed cutoff | Prioritize phase-dynamics diagnostics and transparent negative-result analysis. |
| DelftBlue queue time becomes a hard blocker after using the confirmed share | Activate the DAIC backup path and/or ask DHPC/Mattermost about queue behavior with evidence from `squeue` / `sacct`. |

## Change log

| Date | What changed | Why it matters | Evidence |
|---|---|---|---|
| 2026-05-20 | Added fallback triggers, lane-based workstreams, and claim ledger. | Makes the 12/10 plan visible and reviewable. | `docs/progress.md` |
| 2026-05-20 | Added dataset rule, compute gates, and improvement comparability constraints. | Prevents vague “run more experiments” drift. | `docs/experiment-matrix.md` |
| 2026-05-20 | Added report skeleton and figure/table placeholders. | Forces each experiment to fill a report slot. | `report/README.md` |
| 2026-05-20 | Added RoBERTa-Large feasibility config and Make targets. | Creates a safe gate before cluster-scale reproduction. | `experiments/configs/roberta_large_feasibility.yaml`, `Makefile` |
| 2026-05-20 | Added diagnostics summary mode. | Starts the phase-dynamics evidence path before richer supplement instrumentation. | `scripts/summarize_supplement.py` |
| 2026-05-25 | Removed "waiting for TAs" cluster gating from all setup/plan/progress docs; reframed cluster status as access-available, integration-pending. | Aligns project docs with the actual cluster reality so reproduction sweeps can start as soon as real Slurm templates land. | `docs/setup/delftblue.md`, `slurm/README.md`, `slurm/*.sbatch`, `docs/kickoff.md`, `docs/plans/12-10-paper-track-rolora.md`, `README.md`, `AGENTS.md`, `Makefile` |
| 2026-05-25 | Patched FederatedScope `GPUManager` to fall back to Apple MPS and made the supplement trainer skip fp16 on non-CUDA devices. | Unblocks the RoBERTa-Large feasibility config on a Mac for pre-cluster pipeline checks. | `code/harness/rolora-supplement/RoLoRA-code/federatedscope/core/gpu_manager.py`, `code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/trainer/trainer.py`, `results/roberta_large_feasibility_rolora.log` |
| 2026-05-25 | Authored four QNLI reproduction configs (cells C1-C4), reshaped the experiment matrix around 4 cells × 3 methods × 3 seeds, and disclosed the FlexLoRA omission. | Pins the actual paper-cell hyperparameters (rank 4 for 3/20/50-client; rank 8 only for 50-client; fp32; authors' lr/wd/alpha/dropout) so cluster submissions can start the moment real Slurm headers land. | `experiments/configs/repro_qnli_c{3,20,50}_r{4,8}.yaml`, `docs/experiment-matrix.md`, `docs/progress.md` |
| 2026-05-25 | Added `SEED` env var support to `scripts/smoke_supplement.sh`. | Makes multi-seed runs first-class so per-cell std bars are obtainable without per-seed config duplication. | `scripts/smoke_supplement.sh` |
| 2026-05-25 | Authored real DelftBlue sbatch scripts for the C2 worked example (3 modes × `gpu-a100-small`), wired wandb live tracking into the supplement trainer, auto-derived `LOG_PREFIX` from config basename, and added wandb to the supplement venv install. | Closes the "no script can actually submit a real reproduction job" gap; gives live convergence visibility on every run; lets the 36-job sweep land cleanly once C2 verifies. | `slurm/repro_qnli_c20_r4_{lora,ffa_lora,rolora}.sbatch`, `code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/trainer/trainer.py`, `scripts/smoke_supplement.sh`, `scripts/install_supplement.sh`, `docs/setup/delftblue.md`, `slurm/README.md` |
| 2026-05-25 | Wrote a dum-dum step-by-step DelftBlue runbook (sync → ssh → install → submit). Added login-node policy banner explicitly enumerating what's allowed (`uv sync`, `pip install`, light import checks) vs forbidden (model loads, training, anything GPU). | Encodes the cluster operating constraints so neither the team nor future agents try to run heavy Python on the login node. | `docs/setup/delftblue.md`, `slurm/README.md` |
| 2026-05-25 | Authored `scripts/sync_to_delftblue.sh` (laptop → cluster rsync). Deleted the imported `sbatch_examples/` directory after extracting useful patterns. | One-command sync from laptop. Repo no longer carries example sbatch from another project that didn't match our workload shape. | `scripts/sync_to_delftblue.sh` |
| 2026-05-25 | DelftBlue first-submit discovered three hard constraints: `gpu-a100-small` caps `--mem-per-cpu` at 8000 MB (8G rejected), caps `--cpus-per-task` at 2 (4 rejected), and FederatedScope's `eval.count_flops` flag pollutes the CUDA caching allocator (CUBLAS_STATUS_ALLOC_FAILED on 10 GB MIG). Fixed all three. | Cluster submission now passes scheduler validation AND fits in the 10 GB MIG VRAM budget. We didn't need to escalate to the full `gpu-a100` partition (preserves queue priority). | `slurm/repro_qnli_c20_r4_*.sbatch`, `experiments/configs/repro_qnli_c*.yaml` |
| 2026-05-25 | DelftBlue compute nodes have NO outbound internet (`[Errno 101] Network is unreachable` for huggingface.co). Replaced the slurm-based warm-cache job with `scripts/warm_caches.sh` (login-node, `huggingface_hub.snapshot_download` for files-only download — no model load). Added `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` / `HF_DATASETS_OFFLINE=1` to training sbatch so HF doesn't try sneaky revision-check HEAD requests at job start. | Ground truth of compute-node networking encoded in code and docs. Future agents won't re-litigate this. | `scripts/warm_caches.sh`, `slurm/repro_qnli_c20_r4_*.sbatch`, `docs/setup/delftblue.md`, `slurm/README.md` |
| 2026-05-25 | Refactored wandb logging: REMOVED the trainer-level `wandb.log` (which mixed all clients onto a single noisy zigzag trace), ADDED per-client logging in `Client.callback_funcs_for_evaluate` using FederatedScope's stable `self.ID` (`client_NN/test_acc` etc.), ADDED server-aggregated logging in `Server.merge_eval_results_from_all_clients` using `Results_weighted_avg` (`server/test_acc` etc.). Verified end-to-end: wandb run summary shows 9 `server/*` keys (paper-comparable) + 27 `client_NN/*` keys (per-client diagnostic). | The wandb dashboard now produces the Figure-3-style convergence curve directly (`server/test_acc` over rounds) AND per-client traces for diagnostic. Replaces the previous incorrect logging that conflated all clients onto one trace. | `code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/trainer/trainer.py`, `code/harness/.../federatedscope/core/workers/client.py`, `code/harness/.../federatedscope/core/workers/server.py` |
| 2026-05-25 | Set `WANDB_ENTITY=scalable-learning-7` in all training sbatch and verified a live wandb run lands under the right team. | The team's shared wandb workspace becomes the live evidence channel for every cluster job. | `slurm/repro_qnli_c20_r4_*.sbatch`, run at `https://wandb.ai/scalable-learning-7/sls-rolora-repro` |
| 2026-05-27 | Empirical 6-arm matrix + paper-reading sharpening of ADR 0006. All three modes (rolora/lora/ffa_lora) reach 0.86-0.88 test_acc with AdamW lr=5e-4 in 40 rounds on RoBERTa-base QNLI; all three sit at 0.49-0.52 (chance) with the shipped SGD lr=0.005. Classifier-freeze control arm reached 0.8688 — within 0.01 of unfrozen — confirming the freeze is a minor code-quality bug, not the killer. Paper read (Table 6, p41; Section 5, p7): the paper does NOT specify the optimizer or LR in any table; says only that LR was tuned over `{5e-4,…,1e-1}` with best-of-sweep reported. Daniel's `repro_qnli_c20_r4_*.sbatch` cluster cells additionally ran 30 rounds at 20 clients, but paper's per-client scaling implies ~75 rounds for that setting (two compounding under-tuning problems). | Cluster recipe is decided: (a) AdamW lr 5e-4 + (b) per-paper round count (75 / 30 / 500 for 20 / 50 / 3 clients). The shipped-supplement-doesn't-reach-paper-numbers finding is now a defensible "Reproducibility audit" section in the final report rather than a side note. | `docs/decisions/0006-supplement-reproducibility-gap.md`, `results/overnight_{rolora,lora,ffa_lora}_{sgd,adamw}.log`, `results/overnight_control_originalfreeze_40.log` |
| 2026-05-27 | Matrix runner bug fixes. (a) `scripts/run_supplement_arm.sh` line 67: `${OVERRIDES[@]}` is unbound under `set -u` when no overrides are passed — replaced with the `${OVERRIDES[@]+"${OVERRIDES[@]}"}` idiom, so arms with no overrides (the 3 SGD baselines) no longer crash before training. (b) FederatedScope auto-generates `exp/<auto_name>/sub_exp_<UTC-seconds>` directories at 1-second resolution; two arms launched simultaneously by the matrix collided on the same dir and the second died with FileExistsError — added an explicit `outdir exp/<TAG>_<PID>_<nanosec>` override per arm. Added `scripts/rerun_failed_matrix_arms.sh` to recover the four arms lost the first time. | Without these fixes the overnight matrix only produced 2 of 6 arms on its first dispatch (lost all 3 SGD arms to the unbound variable and `lora_adamw` to the outdir collision). | `scripts/run_supplement_arm.sh`, `scripts/rerun_failed_matrix_arms.sh` |
| 2026-05-27 | Trainer correctness fix + AdamW vs SGD diagnosis. (a) Moved alternation block + `step_count++` inside the `cur_mode in [TRAIN, FINETUNE]` guard and before optimizer construction — upstream code re-fired the block in val/test, drifting `step_count` 3× per round and re-flipping `requires_grad` on a counter that didn't track training rounds. (b) Gated the wandb mech probe to client #1 so `share_local_model=True` doesn't overwrite the start-of-round probe with mid-round client mutations. (c) Added opt-in `SLS_DEBUG_PROBE` / `SLS_DEBUG_GRAD` stdout probes for future diagnostics. (d) Ran a 40-round `rolora` arm with AdamW lr=5e-4 locally on RoBERTa-base QNLI → test_acc 0.8766 by round 39 (started at 0.5054). | Cluster's stuck-at-chance was an **SGD-undertraining problem**, not the classifier-unfreeze fix being wrong. RoLoRA alternation is exact (DBG grad probe: A.grad=None in B-rounds, B.grad=None in A-rounds). Cluster sbatch scripts inherit the authors' SGD lr=0.005 and need to be re-launched with AdamW (or the SGD lr lifted ~10×) before any C2 evidence is meaningful. | `code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/trainer/trainer.py`, `code/harness/.../federatedscope/core/workers/client.py`, `results/overnight_adamw_40.log`, `results/overnight_smoke_final.log` |
| 2026-06-01 | Recorded the confirmed DelftBlue access share: portal display `Education-EEMCS-MSc-DSAIT`, Slurm account directive `education-eemcs-msc-dsait`, with possible one-working-day Slurm accounting lag. | Removes the last account/share ambiguity before the first real C2 submission; prevents future agents from changing the working lower-case `#SBATCH --account` value just because the portal email is capitalized. | DHPC completion email; `docs/setup/delftblue.md`, `slurm/README.md`, `slurm/repro_qnli_c20_r4_*.sbatch`, `AGENTS.md` |
| 2026-06-03 | Reframed the June-11 draft target as compute-constrained reproduction + proxy-scale improvement, then generated report-ready W&B proxy and toy heterogeneity figures. | Gives the draft concrete figures/tables now without pretending the current W&B runs are RoBERTa-Large Table-1 reproduction. | `scripts/plot_wandb_proxy.py`, `scripts/plot_toy_heterogeneity.py`, `evidence/wandb_qnli_c50_r4_20260603/figures/`, `evidence/toy_heterogeneity_20260603/figures/`, `.omx/plans/course-deliverables-results-plan-20260603.md` |
| 2026-06-03 | Added opt-in orthogonal-A initialization, env-gated A/B LoRA learning-rate param groups, and a FedOpt proxy config. | Makes all three proposal improvement axes executable now while preserving default baselines. The real supplement smoke `smoke_improve_orth_ab` verifies orthogonal-A and A/B LR hooks together. | `code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/model/adapter_builder.py`, `code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/trainer/{trainer.py,sls_lora_lr.py}`, `tests/test_sls_{orthogonal_lora_init,lora_lr_groups}.py`, `experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2*.yaml`, `results/overnight_smoke_improve_orth_ab.log` |

## Workstreams

Owners stay as lanes until the team maps names to work. This keeps the setup
usable now without pretending ownership is decided.

| Workstream | Lane owner | Current status | Next action | Evidence path | Blocker / risk |
|---|---|---|---|---|---|
| Infrastructure & baselines | Setup lane | Local env, supplement install path, tests, smoke runs, and pilot summaries exist. RoBERTa-Large feasibility verified locally on Apple MPS. | Run `make table1-medium-all` if local runtime is acceptable; otherwise jump straight to cluster R3/R4/R5/R6. | `experiments/ledger/README.md`, `results/{table1_,roberta_large_feasibility_}*.log` | None blocking. |
| Harness escalation | Setup lane | C2 cluster pipeline staged end-to-end: configs, sbatch (3 modes, partition-compliant), warm-cache login-node script, wandb live logging (`server/*` + `client_NN/*`), runbook. Verified local wandb run lands under `scalable-learning-7/sls-rolora-repro`. | Submit C2 jobs on DelftBlue (one mode × seed 0 first; if green, fan to all 3 × 3). | `slurm/repro_qnli_c20_r4_*.sbatch`, `scripts/warm_caches.sh`, `docs/setup/delftblue.md`, `https://wandb.ai/scalable-learning-7/sls-rolora-repro` | Real cluster wall-time and convergence behaviour still unobserved. |
| Algorithm & ablations | Algorithm lane | RoLoRA/LoRA/FFA-LoRA modes run through the patched supplement; invariant tests exist for local helper code. | Add/verify active-factor update norms and frozen-factor markers before serious runs. | `code/harness/rolora-supplement.patch`, `tests/` | Supplement code is gitignored; patch discipline matters. |
| Improvement & analysis | Analysis lane | Orthogonal-A has 5-seed toy signal (+3.30 pp); orthogonal-A proxy seed 0 is running; A/B LR and FedOpt are code/config-ready. | Monitor `results/overnight_proxy_orth_a_c50_r4_lr1e-2_seed0.log`; if competitive, run seeds 1/2, otherwise try A/B LR seed 0. | `report/draft_results_brief_20260603.md`, `evidence/toy_heterogeneity_20260603/figures/`, `experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2*.yaml` | GLUE/proxy improvement result still pending; toy improvement must be labelled toy-only. |
| Report & presentation | Analysis lane | Paper presentation outline exists; report has a claim-led skeleton and a June-11 draft results brief. | Draft prose from existing W&B proxy + toy improvement evidence while the orthogonal-A proxy run finishes. | `report/README.md`, `report/draft_results_brief_20260603.md`, `docs/templates/paper-presentation-outline.md` | Writing too late will make results look like an experiment dump. |
| Cluster / access | Setup lane | DelftBlue full access granted 2026-06-01 under portal share `Education-EEMCS-MSc-DSAIT`; templates use Slurm account `education-eemcs-msc-dsait`. Daniel's first cluster attempts are now evidence that the old shipped SGD/30-round recipe was under-tuned, not evidence against RoLoRA. `main` now carries the corrected AdamW/75-round C2 recipe plus trainer fixes. | Sync `main`, verify Slurm sees the confirmed account, relaunch corrected `repro_qnli_c20_r4_rolora.sbatch` seed 0, and ledger the result. | `docs/setup/delftblue.md`, `slurm/README.md`, `slurm/repro_qnli_c20_r4_*.sbatch`, `scripts/{sync_to_delftblue,warm_caches}.sh`, `docs/decisions/0006-supplement-reproducibility-gap.md` | Slurm accounting may lag the 2026-06-01 grant by up to one working day. Risk: 10 GB MIG VRAM/4h wall time may truncate the 75-round run; partial ≥30-round AdamW trajectory is still useful, full completion may need `gpu-a100`. |

## Claim ledger

Every report claim must map to evidence. Keep unsupported claims in `planned`,
not in prose.

| Claim ID | Claim | Status | Required evidence | Config / command | Seeds | Log / plot | Owner | Reviewer | Limitations |
|---|---|---|---|---|---|---|---|---|---|
| C0 | The local harness preserves RoLoRA/LoRA/FFA-LoRA execution modes. | supported-local | Smoke logs with `[sls-rolora]` markers. | `make supplement-smoke-all` | 0 | `results/smoke_*.log` | Setup lane | TBD | Pipeline evidence only. |
| C1 | The local toy reproduces the qualitative Figure-2 ordering. | supported-local | MNIST plot and final accuracies. | `make mnist-paper` | 0 | `results/mnist_fig2.png` | Analysis lane | TBD | Toy model, not GLUE. |
| C2 | RoLoRA is comparable to or better than LoRA/FFA-LoRA at the local Table-1-shaped rung. | running | Medium all-mode summary. | `make table1-medium-all && make table1-medium-summary` | 0 | `results/table1_medium_*.log` | Setup lane | TBD | RoBERTa-base/QNLI only. |
| C3 | RoBERTa-Large feasibility is known before cluster spend. | supported-local | One tiny feasibility run or actionable failure. | `make roberta-large-feasibility MODE=rolora` (CUDA / Apple MPS / CPU) | 0 | `results/roberta_large_feasibility_rolora.log` | Setup lane | TBD | Verified on Apple MPS; cluster feasibility-equivalent re-verification will fall out of the first C2 submission. |
| C4 | RoLoRA degrades less than LoRA/FFA-LoRA as clients increase. | planned | 3/20/50-client (r=4) and 50-client (r=8) RoBERTa-Large table and Figure-3-style 50-client curve, all on QNLI. | R3-R6 matrix rows; configs `experiments/configs/repro_qnli_c{3,20,50}_r{4,8}.yaml`; C2 sbatch scripts ready under `slurm/repro_qnli_c20_r4_*.sbatch`; wandb dashboard `sls-rolora-repro` is the live evidence channel. | 0,1,2 per cell | TBD (wandb + `results/*.log` + `exp/*/sub_exp_*/eval_results.log`) | Algorithm lane | TBD | FlexLoRA omitted (not in supplement). |
| C5 | Phase-specific diagnostics explain at least one improvement or null result. | planned | Per-round phase, update norm, metric, and wall-time traces. | I1-I5 matrix rows | 0 first, replicate winner | TBD | Analysis lane | TBD | Requires supplement instrumentation beyond current final metrics. |
| C6 | At RoBERTa-base proxy scale, the current 50-client QNLI W&B bundle gives a usable RoLoRA control and LR choice for improvement transfer. | supported-proxy | W&B-exported server curves and summary table; 28 runs / 560 server-history rows. | `uv run python scripts/plot_wandb_proxy.py` | RoLoRA `1e-2`: 0,1,2; other LR/methods as exported | `evidence/wandb_qnli_c50_r4_20260603/figures/proxy_plot_summary.md`, `proxy_a_*`, `proxy_b_*`, `proxy_c_*`, `proxy_d_*`, `proxy_e_*` | Analysis lane | TBD | RoBERTa-base proxy only; not Table 1; all W&B states are `crashed` after completed rounds 0-19; run metadata lacks full config provenance. |
| C7 | In the extreme toy heterogeneity setting, RoLoRA beats LoRA/FFA-LoRA and orthogonal-A initialization improves RoLoRA. | supported-toy | Daniel branch JSON results copied into evidence and replotted with 5-seed CI95. | `uv run python scripts/plot_toy_heterogeneity.py` | 0,1,2,3,4 | `evidence/toy_heterogeneity_20260603/figures/toy_plot_summary.md`, `toy_g_*`, `toy_h_*` | Analysis lane | TBD | Toy MNIST model; improvement still needs GLUE/proxy transfer before final strong claim. |
| C8 | Orthogonal-A initialization is implemented as an opt-in PEFT-LoRA initializer without changing the default baseline. | running-proxy | Unit test on dummy LoRA-A/LoRA-B parameters; py_compile of adapter builder; real supplement smoke; matched proxy seed-0 run in progress. | `SLS_LORA_INIT=orthogonal_a`; `experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml` | smoke; proxy seed 0 running | `tests/test_sls_orthogonal_lora_init.py`, `results/overnight_smoke_improve_orth_ab.log`, `results/overnight_proxy_orth_a_c50_r4_lr1e-2_seed0.log` | Algorithm lane | TBD | Toy improvement supported; GLUE/proxy result pending until current run completes. |
| C9 | Separate LoRA-A / LoRA-B learning rates are implemented as an opt-in improvement axis without changing default baselines. | supported-code | Unit test for default-noop and env-gated param groups; real supplement smoke confirms phase-specific groups. | `SLS_LORA_LR_A=0.005 SLS_LORA_LR_B=0.01` | n/a | `tests/test_sls_lora_lr_groups.py`, `results/overnight_smoke_improve_orth_ab.log` | Algorithm lane | TBD | No result claim yet; run after orthogonal-A seed 0 if needed. |
| C10 | Adaptive server-side optimization is available via FederatedScope FedOpt for the same proxy cell. | supported-config | Config syntax check; FederatedScope `FedOptAggregator` is wired by `fedopt.use True`. | `experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2_fedopt_adam.yaml` | n/a | config parse validation; `fedopt.optimizer.type=Adam`, `fedopt.optimizer.lr=0.1` | Algorithm lane | TBD | No result claim yet; first run should be seed 0 only. |

## This-week checklist

- [x] Make the 12/10 plan visible in README/progress/matrix/report docs.
- [x] Add the RoBERTa-Large feasibility config and command target.
- [x] Add diagnostics summary parsing for existing supplement logs.
- [x] RoBERTa-Large feasibility verified locally on Apple MPS.
- [x] Author the four QNLI reproduction configs (C1-C4) using authors' `test_glue.yaml` hyperparameters.
- [x] Author three C2 sbatch files (lora / ffa_lora / rolora) for `gpu-a100-small`, partition-compliant.
- [x] Stand up `scripts/sync_to_delftblue.sh` (laptop → cluster rsync) and `scripts/warm_caches.sh` (login-node model+dataset download).
- [x] Refactor wandb logging: per-client traces using FederatedScope's `self.ID` plus server-aggregated weighted-avg trace.
- [x] Write the dum-dum DelftBlue runbook with login-node policy banner.
- [x] Generate W&B proxy figures from the 28-run QNLI/C50/r4/RoBERTa-base bundle for the June-11 draft.
- [x] Copy and replot Daniel's 5-seed toy heterogeneity + orthogonal-A results.
- [x] Implement the opt-in orthogonal-A GLUE proxy transfer switch and proxy config.
- [x] Add A/B LoRA LR groups and FedOpt proxy config so all proposal axes are command-ready.
- [ ] Monitor the running orthogonal-A GLUE proxy transfer against the RoLoRA `lr=1e-2` control (`results/overnight_proxy_orth_a_c50_r4_lr1e-2_seed0.log`). **← draft-critical next action.**
- [ ] Submit one RoBERTa-Large attempt only after the proxy/improvement draft figures are wired, or ledger a clean hardware/runtime blocker.
- [ ] Keep full R3-R6 Table-1 reproduction as final/stretch work, not as a June-11 blocker.
- [ ] Run `make table1-medium-all` (now optional — paper-scale is unblocked on cluster, local pilot is supplementary).
- [ ] Record every experiment attempt, including failures, in `experiments/ledger/README.md`.
- [ ] Map human names to setup / algorithm / analysis lanes.

## Evidence rules

- Every result claim needs a command, config, seed, log/plot path, and interpretation.
- Local RoBERTa-base pilot metrics are pipeline evidence only; do not compare them directly to paper Table 1.
- Failed runs are evidence. Keep the log and record the blocker.
- No improvement claim is report-ready unless it has a baseline, a curve, and a phase-dynamics explanation.

## 2026-06-03 — local M4/MPS improvement execution update

- Confirmed local PyTorch MPS availability in both the project `uv` env and the RoLoRA supplement venv.
- The active orthogonal-A QNLI proxy run is still running locally:
  - log: `results/overnight_proxy_orth_a_c50_r4_lr1e-2_seed0.log`
  - W&B group: `qnli_c50_r4_improvements`
  - current diagnostic extraction: `evidence/improvement_diagnostics_20260603/proxy_orth_a_seed0_partial/`
- Early server diagnostics now show phase sensitivity:
  - round 0 B: `test_acc=0.509793`
  - round 1 A: `test_acc=0.509061`
  - round 2 B: `test_acc=0.563793`
  - round 3 A: `test_acc=0.506864`
  - round 4 B is pending server aggregation but client mean is already ~`0.722724`.
- Added and queued the next real experiment after the active run exits:
  - `SLS_LORA_INIT=orthogonal_a SLS_PHASE_PATTERN=BBA SLS_DEVICE=mps`
  - tag/log: `proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed0`, `results/overnight_proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed0.log`
  - queue log: `results/queue_proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed0.log`
- Verification after device/controller/diagnostic changes:
  - `uv run pytest -q tests/test_sls_phase_schedule.py tests/test_sls_orthogonal_lora_init.py tests/test_sls_lora_lr_groups.py` → 11 passed.
  - `uv run ruff check ...` → passed.
  - `bash -n scripts/run_supplement_arm.sh scripts/smoke_supplement.sh results/run_queued_proxy_phase_bba_orth_a_seed0.sh` → passed.
  - `py_compile` for modified supplement modules → passed.
  - `git diff --check` → passed.

## 2026-06-03 — monitor/plot instrumentation update

- Added live internal monitors for improvement runs and verified them on the M4/MPS supplement path:
  - trainer start/end factor norms: `||A||`, `||B||`, `||BA||`, classifier norm;
  - local update norms: `ΔA`, `ΔB`, `Δclassifier`;
  - aggregation drift/update norms when a round reaches server aggregation.
- Active monitored run:
  - command shape: `SLS_LORA_INIT=orthogonal_a SLS_PHASE_PATTERN=BBA SLS_DEVICE=mps SLS_MONITOR=1 MODE=rolora`
  - log: `results/overnight_proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed0.log`
  - W&B run: `https://wandb.ai/scalable-learning-7/sls-rolora-repro/runs/8a2yoamg`
  - current evidence: `evidence/improvement_diagnostics_20260603/proxy_phase_bba_orth_a_seed0_partial/`
- New plots are generated by `scripts/plot_supplement_diagnostics.py`:
  - `figures/supplement_diagnostics_curve.{png,pdf}` for accuracy/client-spread/phase bands;
  - `figures/supplement_internal_monitors.{png,pdf}` for factor norms and update norms.
- Current BBA monitor extract has round-0 local rows before server aggregation: `internal monitor rows=61`, `train_n=30`, mean `||BA||=0.017690`, mean `ΔA=0.000000`, mean `ΔB=0.017690`, mean `Δclassifier=0.030826`.
- Important limitation: the earlier orthogonal-A run was launched before `SLS_MONITOR=1`, so it has outcome-level diagnostics only. It cannot retroactively gain internal monitor rows; compare its accuracy curve against the monitored BBA run once BBA reaches server evals.
- Verification after extractor/plot changes:
  - `uv run pytest -q tests/test_sls_phase_schedule.py tests/test_sls_orthogonal_lora_init.py tests/test_sls_lora_lr_groups.py` → 11 passed.
  - `uv run ruff check scripts/extract_supplement_metrics.py scripts/plot_supplement_diagnostics.py scripts/plot_wandb_proxy.py scripts/plot_toy_heterogeneity.py tests/test_sls_phase_schedule.py tests/test_sls_orthogonal_lora_init.py tests/test_sls_lora_lr_groups.py` → passed.
