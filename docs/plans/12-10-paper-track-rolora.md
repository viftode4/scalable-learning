# Push the RoLoRA final project from solid reproduction to 12/10 paper-track work

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

## Purpose / Big Picture

The goal is to turn the current CS4725 RoLoRA project from a well-organized reproduction attempt into a top-grade, professor-interesting research project. A merely good project reproduces some RoLoRA results and tries one improvement. A 12/10 project makes a clear scientific claim, proves the claim with a credible reproduction baseline, runs controlled ablations, explains failures honestly, and packages the work so the course staff can see workshop-paper potential.

The observable outcome is a week-9 final report and presentation where every major claim cites a command, config, seed, log, and plot; where the improvement story has one primary thesis rather than three disconnected tweaks; and where the team can show a clean artifact trail from the original RoLoRA paper to local sanity checks, paper-scale reproduction cells, and improvement experiments.

## Progress

- [x] (2026-05-20 18:55 Europe/Amsterdam) Reviewed local paper, proposal, lecture rubric, repo README, kickoff TODOs, and experiment ledger.
- [x] (2026-05-20 18:55 Europe/Amsterdam) Confirmed current repo has local evidence only, not paper-comparable RoBERTa-Large reproduction yet.
- [x] (2026-05-20 18:55 Europe/Amsterdam) Created this 12/10 paper-track plan under `docs/plans/12-10-paper-track-rolora.md`.
- [ ] Assign named owners for infrastructure/baselines, algorithm/ablations, and improvement/analysis.
- [x] (2026-05-20) Promoted `docs/progress.md` into the single live progress board with fallback triggers and a claim ledger.
- [ ] Run and log `make table1-medium-all`, then add results to `experiments/ledger/README.md`.
- [x] (2026-05-20) Added a RoBERTa-Large one-round feasibility config and Make target; GPU run still pending.
- [x] (2026-05-20) Decided the unified phase-specific A/B dynamics thesis and recorded it in ADR 0005.
- [x] (2026-05-20) Started the claim-led final report skeleton in `report/README.md`.
- [x] (2026-05-20) Added diagnostics summary parsing for supplement logs.
- [x] (2026-05-25) Patched supplement for Apple MPS fallback and CPU/MPS fp16 guard; RoBERTa-Large feasibility now runs on Mac as a pre-cluster sanity path.
- [x] (2026-05-25) Removed all "waiting for TAs" cluster gating from setup/plan/progress/README docs; reframed as access-available, integration-pending.
- [x] (2026-05-25) Authored four QNLI reproduction configs (cells C1-C4 = 3, 20, 50r4, 50r8 clients) using authors' `test_glue.yaml` hyperparameters (fp32, lr=0.005, wd=0.0002, alpha=32, dropout=0.1, 30 rounds × 20 local batches × bs=32 × tok_len=128). Reshaped the experiment matrix around 4 cells × 3 methods × 3 seeds = 36 jobs per dataset. Disclosed FlexLoRA omission (not in supplement).
- [x] (2026-05-25) Added `SEED` env var support to `scripts/smoke_supplement.sh` and auto-derived `LOG_PREFIX` from config basename so per-seed logs name themselves correctly.
- [x] (2026-05-25) Authored three real DelftBlue sbatch scripts for the C2 worked example (`slurm/repro_qnli_c20_r4_{lora,ffa_lora,rolora}.sbatch`). Made them partition-compliant on `gpu-a100-small` (cpus-per-task=2, mem-per-cpu=8000M, time=03:59:59) after the first-submit attempts surfaced the per-partition caps. Added `eval.count_flops: False` to all four repro YAMLs to avoid the FlopCountAnalysis CUDA-cache pollution that caused `CUBLAS_STATUS_ALLOC_FAILED` on the 10 GB MIG slice. Set `WANDB_ENTITY=scalable-learning-7`. Added `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` / `HF_DATASETS_OFFLINE=1` to all training sbatch.
- [x] (2026-05-25) Compute-node networking reality discovered (`[Errno 101] Network is unreachable` for `huggingface.co` from `gpu*` nodes). Replaced the slurm warm-cache job with `scripts/warm_caches.sh` (login-node, uses `huggingface_hub.snapshot_download` for file-fetch-only — no model load — so it fits the "install script or two" login-node policy).
- [x] (2026-05-25) Wandb logging refactored to be paper-aligned. Trainer's per-routine `wandb.log` was producing a noisy single-trace zigzag mixing all clients. Replaced with: per-client logging in `Client.callback_funcs_for_evaluate` using FederatedScope's stable `self.ID` (keys `client_NN/{metric}`), plus server-aggregated logging in `Server.merge_eval_results_from_all_clients` using `Results_weighted_avg` (keys `server/{metric}`). Verified locally: wandb run summary shows 9 `server/*` keys (paper-comparable convergence) + 27 `client_NN/*` keys (3-client local feasibility) across `scalable-learning-7/sls-rolora-repro`.
- [x] (2026-05-25) Wrote the dum-dum DelftBlue runbook (`docs/setup/delftblue.md`) with explicit login-node policy banner and 0-9 step checklist. Authored `scripts/sync_to_delftblue.sh` for laptop → cluster rsync.

## Surprises & Discoveries

- Observation: The current repo is stronger on engineering hygiene than on research-control hygiene.
  Evidence: `README.md` defines layout, commands, and status; `docs/decisions/` has ADRs; `experiments/ledger/README.md` records evidence. But `docs/kickoff.md` still has TODOs for owner mapping, tech lead, cadence, Git workflow, and paper-presentation roles.

- Observation: The proposal commits to three improvement directions, while the deep-research plan’s strongest paper-shaped angle is partial participation. This is a strategic tension.
  Evidence: `README.md` and `docs/research/project-proposal.pdf` commit to improved initialization, separate A/B learning rates, and adaptive server-side optimization. `docs/research/deep-research-plan.md` recommends partial participation as the strongest paper niche, but project guidance says that recommendation is superseded by the submitted proposal.

- Observation: The current local Table-1-shaped pilot is not a scientific result yet.
  Evidence: `experiments/ledger/README.md` explicitly says the RoBERTa-base QNLI local pilot is too tiny for paper-comparable numbers, and `README.md` says full Table 1 requires RoBERTa-Large across GLUE tasks/client counts/multiple seeds.

- Observation (2026-05-25): The authors' supplement omits FlexLoRA entirely despite reporting it in Table 1. Their `test_glue.yaml` covers only LoRA / FFA-LoRA / RoLoRA via the alternation switch. We reproduce three methods, not four; this is disclosed as a known limitation in the report.
  Evidence: `grep -rn FlexLoRA code/harness/rolora-supplement/RoLoRA-code/` returns zero matches. Decision documented in `docs/experiment-matrix.md` "Methods scope and FlexLoRA gap" section.

- Observation (2026-05-25): DelftBlue's `gpu-a100-small` partition has narrower per-knob caps than its `gpu-a100` counterpart — `--mem-per-cpu` capped at 8000 MB (not 8 GiB), `--cpus-per-task` capped at 2, ≤4h wall time. None of these were documented in the partition-info we inherited; all three surfaced as `sbatch: error: ...` on first submission and were fixed in place.
  Evidence: `slurm/repro_qnli_c20_r4_*.sbatch` headers; progress.md 2026-05-25 change-log row "DelftBlue first-submit discovered three hard constraints".

- Observation (2026-05-25): DelftBlue compute nodes have NO outbound internet for `huggingface.co` (`[Errno 101] Network is unreachable`). This invalidated the original "compute-node warm-cache" sbatch design. Model and dataset downloads must happen on the login node; the runbook now uses `scripts/warm_caches.sh` with `huggingface_hub.snapshot_download` (file-fetch only — no model load — so it fits the "install script or two" login-node policy). Wandb's specific endpoint *is* reachable from compute nodes (verified empirically), so live metrics streaming still works.
  Evidence: `slurm_logs/sls-warm-caches-9971216.err` (the original network-unreachable trace); `docs/setup/delftblue.md` policy banner.

- Observation (2026-05-25): FederatedScope's default `eval.count_flops: True` triggers `fvcore.nn.FlopCountAnalysis` on the first batch, which leaks CUDA caching-allocator blocks and causes `CUBLAS_STATUS_ALLOC_FAILED` on the 10 GB MIG slice — the model itself fits, but FLOPS counting tips it over. The authors' supplement even prints a warning about this. We set `count_flops: False` in every reproduction config. The FLOPS number is never reported in paper Table 1 or Figure 3, so the loss is zero.
  Evidence: training log showing `WARNING: When using count flops functions, ...` then `CUBLAS_STATUS_ALLOC_FAILED`; `experiments/configs/repro_qnli_c*.yaml` all have `count_flops: False`.

- Observation (2026-05-25): FederatedScope's wandb-friendly logging surface is essentially "look at `eval_results.log` after the fact." The authors shipped zero plotting / aggregation tools. The right architecture for paper-comparable live tracking is to hook wandb into `Server.merge_eval_results_from_all_clients` (for `Results_weighted_avg`, the paper number) and `Client.callback_funcs_for_evaluate` (for per-client diagnostic traces with FederatedScope's stable `self.ID`). The trainer-level wandb hook, originally added, was wrong because trainer is per-client without stable identity.
  Evidence: `code/harness/rolora-supplement/RoLoRA-code/federatedscope/core/workers/{client,server}.py` patches; `https://wandb.ai/scalable-learning-7/sls-rolora-repro/runs/0ybxxdta` verifies 9 `server/*` + 27 `client_NN/*` keys live.

## Decision Log

- Decision: Treat the course proposal’s three improvement directions as the official scope, but organize them under one thesis: “RoLoRA is sensitive to the phase-specific geometry and optimizer dynamics of A versus B.”
  Rationale: This preserves the proposal while avoiding three disconnected experiments. Initialization probes the geometry of A at round 0; separate learning rates probe phase-specific local optimization; adaptive server optimization probes how aggregated active-factor updates should be accumulated across rounds.
  Date/Author: 2026-05-20 / Codex

- Decision: Prioritize a narrow, high-quality reproduction over broad but weak coverage.
  Rationale: The course asks to reproduce and improve a top-tier article. A 12/10 report must establish trust first. One dataset with RoBERTa-Large, 3/20/50 clients, all three baselines, several seeds, and convergence curves is stronger than shallow partial runs over five tasks without seed control.
  Date/Author: 2026-05-20 / Codex

- Decision: Make the “paper-track” contribution empirical-first, not theory-first.
  Rationale: With three people and a week-9 deadline, a publishable workshop-style contribution is most likely to come from clean characterization, controlled ablations, and reproducible negative/positive results. New theory can be included as explanatory framing only if time remains.
  Date/Author: 2026-05-20 / Codex

## Outcomes & Retrospective

Current outcome (as of 2026-05-25): the plan identifies the critical gap between the current repository state and a top-grade final project. The repository now has the core research-control surfaces; the C2 cluster pipeline is staged end-to-end (configs, sbatch, warm-cache, wandb live, runbook); the only un-derisked step is the actual first DelftBlue training submission. The remaining 12/10 risk is execution evidence (real cluster results), diagnostics maturity (update-norm / frozen-factor markers beyond final metrics), and named human ownership.

The main lesson is that the project should stop thinking in terms of "run more experiments" and start thinking in terms of "prove one compelling claim with the minimum experiment set that withstands scrutiny." The 2026-05-25 session also surfaced a meta-lesson: cluster-policy knowledge (login-node restrictions, per-partition resource caps, compute-node network policy, FederatedScope's CUDA cache footgun) only crystallises through real submission attempts. We encoded each discovery in code + docs as it happened, which is the right pattern — but it underlines that "the first real sbatch submission" is itself a high-information experiment, not a formality.

## Context and Orientation

This repository is a TU Delft CS4725 project reproducing and extending RoLoRA: “Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA.” RoLoRA is a federated learning method for LoRA adapters. LoRA represents a weight update as the product of two low-rank matrices, commonly called A and B. In federated learning, several clients train locally and a central server aggregates updates. Directly averaging A and B separately does not equal averaging the full product A times B, so ordinary FedAvg over LoRA factors introduces an aggregation error. RoLoRA avoids this by alternating: in one communication round A is frozen and B is trained/aggregated; in the next round B is frozen and A is trained/aggregated. Because one factor is shared and fixed in each round, aggregation is exact for the active factor.

The local source documents are the authority. `docs/research/paper-rolora.pdf` is the paper. `docs/research/project-proposal.pdf` is the submitted proposal. `docs/research/lecture-01-introduction.pdf` gives assessment: paper presentation is 20%, research project is 60%, homeworks are 20%; project deliverables are final report and final presentation in week 9, with draft report in week 8. `README.md` summarizes this in repository form. `experiments/ledger/README.md` records evidence already collected.

The current repo has good engineering structure: `Makefile` commands, `scripts/` runners, `experiments/configs/` YAML configs, `tests/` invariants, `docs/decisions/` ADRs, and local results logs. The research-control structure now has a live progress board, experiment matrix, claim ledger, and report skeleton; the remaining weakness is execution evidence and named human ownership.

## Critical Assessment Against a 12/10 Standard

The current project is good for week 3, but it is not yet a 12/10 project. A 12/10 project must make the grader think: “this team understood the paper deeply, reproduced the central claim credibly, discovered something non-obvious, and packaged it like a small research artifact.” The current state mostly proves that the team can run and patch the harness. That is necessary, but not sufficient.

The biggest weakness is that the improvement plan is currently too diffuse. “Improved initialization,” “separate learning rates,” and “adaptive server-side optimization” are each plausible, but if reported as three independent tweaks they will read like a parameter sweep rather than a contribution. The fix is to unify them around A/B phase dynamics. The report should ask: “RoLoRA’s exact aggregation relies on alternating A and B, but are the two phases equally stable and equally optimized?” Then each improvement becomes part of the same investigation.

The second weakness is that the reproduction target is still underspecified operationally. The proposal says Table 1 and Figure 3. The repo says full Table 1 is too expensive locally. The team needs a graded fallback ladder. Minimum acceptable reproduction for a strong grade is one GLUE dataset at RoBERTa-Large with 3, 20, and 50 clients across LoRA, FFA-LoRA, and RoLoRA, plus convergence curves for the 50-client case. Better is three datasets. Full 5-task Table 1 is ideal but not required if it prevents deeper analysis.

The third weakness is report risk. The report skeleton now exists, but it is still mostly placeholders. If sections are not filled as evidence arrives, the project will still look like an experiment dump. Every experiment should be run because it fills a named slot in the report, not because it is “next.”

The fourth weakness is team risk. `docs/kickoff.md` still has unassigned ownership, no tech lead, no confirmed cadence, and no Git workflow. A three-person project aiming above 10/10 cannot run by vibes. The team needs one owner per layer and one person explicitly responsible for the final story.

## Target Research Thesis

Use this thesis unless the team discovers stronger evidence:

RoLoRA’s advantage comes from exact alternating aggregation, but its performance is governed by phase-specific behavior of A and B. Better initialization of A, asymmetric A/B learning rates, and lightweight server-side adaptation can improve convergence or stability because they target different parts of the same phase-specific bottleneck.

This thesis is attractive because it satisfies the proposal and creates a coherent story. It also gives publishable shape: the contribution is not “we tried three tricks,” but “we characterize phase-specific bottlenecks in RoLoRA and evaluate three minimal interventions that preserve exact aggregation and communication efficiency.”

## Grade Strategy

For a 10/10, the team needs credible reproduction, at least one reasonable improvement, and a clear report. For a 12/10, the team needs all of the following.

First, it must reproduce the core effect, not just run the code. The core effect is that as client count rises, vanilla FedAvg-LoRA degrades more than RoLoRA, and RoLoRA has better convergence behavior in the 50-client setting. The final report must show this with RoBERTa-Large or explain exactly why compute prevented it and provide the strongest possible substitute.

Second, it must include ablations that isolate mechanisms. Do not only report final accuracy. Report convergence speed, early-round stability, variance across seeds, and whether gains occur mainly in A-update rounds or B-update rounds. The professor will be more interested in a diagnostic result than a marginal +0.3% accuracy table.

Third, it must be honest about negative results. If orthogonal init does not improve final accuracy but reduces variance or improves early convergence, that is still a result. If LoRA+ ratios help LoRA but not RoLoRA, that is interesting. If server Adam destabilizes alternating updates, that is useful.

Fourth, it must include reproducibility artifacts. Every table cell should map to a config, command, seed, and log path. The appendix should explain exact code changes to the supplement and how to rerun the main cells.

## Plan of Work

Milestone 1 is to keep project-control structure current. `docs/progress.md`, `docs/experiment-matrix.md`, ADR 0005, and `report/README.md` are the control surfaces. Update them before and after every substantial experiment so evidence remains organized.

Milestone 2 is the local-to-cluster bridge. The local rungs (`make table1-medium-all`, then `make roberta-large-feasibility`) exist to retire pipeline risk before cluster submission. The RoBERTa-Large feasibility config is now also Mac-runnable on MPS, which gives a fast pre-submission sanity check. The acceptance criterion at this milestone is not accuracy; it is that the model loads, the data path works, memory use is known, and the log contains the same method markers as local pilots. Once the user-supplied real DelftBlue `#SBATCH` headers are dropped into `slurm/gpu-a100-small.sbatch` and `slurm/gpu-v100.sbatch`, the same feasibility config is the first cluster submission, and we ledger that result.

Milestone 3 is the minimum credible reproduction and is the active milestone. Run one RoBERTa-Large dataset for 3, 20, and 50 clients across LoRA, FFA-LoRA, and RoLoRA on DelftBlue. The exact dataset (MNLI primary, QNLI fallback) is less important than getting one full sweep through the pipeline end-to-end first; the dataset choice can be refined for subsequent sweeps once the first one lands. Use at least three seeds for the most important 50-client comparison; if compute is constrained, use one seed for 3/20 and three seeds for 50. Generate Figure-3-style convergence curves for 50 clients. The acceptance criterion is a table and curve that directly test the paper’s central client-scaling claim, with every row mapped to a config, command, seed, and log path in `experiments/ledger/README.md`.

Milestone 4 is the improvement characterization. Implement or expose experiment knobs for the three committed directions. Improved initialization should include standard PEFT init, orthogonal A, and one SVD/PCA-like data-informed A if feasible. Separate learning rates should sweep ratios such as A:B equals 1:1, 1:4, 1:8, 1:16, and 4:1 depending on which phase appears undertrained. Adaptive server-side optimization should start with FedAvg versus server momentum or server Adam on the active factor only. The acceptance criterion is not that all three beat RoLoRA; it is that each answers a precise question about phase-specific dynamics.

Milestone 5 is report and presentation hardening. Keep the report skeleton current as evidence arrives. Each figure should have an owner, expected command, and fallback. The paper presentation should focus on RoLoRA itself, but it should preview the project extension in the discussion section. The final project presentation should be story-first: problem, reproduction trust, phase-specific diagnosis, intervention results, limits.

## Concrete Steps

From repository root `/Users/vliftode/personal/scalable-learning`, first verify the baseline:

    uv run pytest -q
    uv run ruff check .

Expect all tests to pass. On 2026-05-20 this was observed as 23 tests passing and Ruff reporting all checks passed.

Maintain the control docs:

    docs/progress.md
    docs/experiment-matrix.md
    report/README.md

`docs/progress.md` is the live dashboard and claim ledger. `docs/experiment-matrix.md` is the run contract with dataset rules, compute gates, and improvement constraints. `report/README.md` is the report skeleton and figure/table inventory.

Run the next local rung:

    make table1-medium-all
    make table1-medium-summary
    make diagnostics-summary PREFIX=table1_medium

Add the outputs to `experiments/ledger/README.md` with date, command, scale, evidence path, and interpretation. If the three local methods have identical accuracy, explicitly write that this is expected or not interpretable at tiny scale; do not overclaim.

Maintain the report skeleton under `report/`. It must keep figure placeholders current: RoLoRA mechanism diagram, MNIST sanity result, reproduction table, 50-client convergence curve, phase-specific improvement ablation, and limitations table.

Before the first cluster submission, drop the user-supplied real `#SBATCH` header and `module load` lines into `slurm/gpu-a100-small.sbatch` and `slurm/gpu-v100.sbatch`, then submit the RoBERTa-Large feasibility config (`experiments/configs/roberta_large_feasibility.yaml`) once per mode. Update `docs/setup/delftblue.md` to remove the "pending" banner only after a feasibility job actually completes on DelftBlue and the result is ledgered.

## Validation and Acceptance

The project is on a 12/10 trajectory only when the following are true.

First, `make check` passes before cluster runs. Second, every experiment claim has a row in `experiments/ledger/README.md`. Third, every major report figure has a config and log path. Fourth, the team can answer: “What is the one scientific claim beyond reproduction?” in one sentence. Fifth, the midterm review can show a live table of done/running/blocked experiments rather than a verbal status update.

The final 12/10 acceptance bar is: the report contains a credible reproduction of RoLoRA’s client-scaling claim, at least one improvement or characterization result that teaches something about A/B alternation, clear negative-result handling, and a reproducibility appendix. The presentation must lead with the problem and contribution, not implementation details.

## Idempotence and Recovery

All documentation steps are safe to repeat. If an experiment fails, do not delete the log. Add it to the ledger as a failed run with the error and next action. If DelftBlue queue time becomes a hard blocker, file a faculty-share / TOPdesk request and/or activate the DAIC backup path; in the meantime, the strongest local substitute is RoBERTa-base longer runs plus the Mac-runnable RoBERTa-Large feasibility probe and deeper MNIST/linear characterization of phase-specific behavior. If the supplement becomes a blocker, switch to the FedSA-LoRA fallback per ADR 0001 and document the switch in a new ADR.

If an improvement direction shows no accuracy gain, recover by reframing it as characterization: report convergence speed, variance, phase sensitivity, and failure mode. A clean negative result with explanation is better than hiding the experiment.

## Artifacts and Notes

Current local evidence already exists in `experiments/ledger/README.md`: MNIST paper sanity, supplement smoke-all, table1 pilot all modes, and table1 medium RoLoRA. Current gaps are explicitly listed in `docs/kickoff.md`: cluster access, time commitment, team split, tech lead, presentation roles, homework collision, cadence, Git workflow, and reproduction milestone sanity-check.

Key source evidence from the local paper: RoLoRA’s paper evaluates RoBERTa-Large on SST-2, QNLI, MNLI, QQP, and RTE; Table 1 varies client counts 3, 20, and 50; Figure 3 shows 50-client convergence curves; the paper assumes all clients participate in each round in the language-model experiments. The submitted proposal says the project aims to reproduce Table 1 and Figure 3 and then evaluate improved initialization, separate learning rates, and adaptive server-side optimization while preserving RoLoRA’s alternating structure.

## Interfaces and Dependencies

Use the existing repository interfaces. `Makefile` is the public command interface. `experiments/configs/` is the configuration interface. `scripts/run_supplement.py` and `scripts/smoke_supplement.sh` are the supplement execution interface. `scripts/summarize_supplement.py` is the first summary interface. `experiments/ledger/README.md` is the evidence interface. New report and progress docs should reference these rather than inventing a parallel workflow.

Do not add new dependencies for project management. Use Markdown tables and existing scripts. Add code only when it directly enables an experiment or validation that appears in the final report.


## External Literature Snapshot Checked on 2026-05-20

This snapshot was added after the user clarified that online research is acceptable when useful. It should be refreshed before finalizing the report, but it is enough to shape the immediate project strategy.

RoLoRA itself is confirmed as a NeurIPS 2025 poster on OpenReview, last modified 2026-04-21, with supplementary material available. The OpenReview abstract emphasizes alternating optimization, learning both projection matrices, reduced communication, and RoBERTa-Large/Llama-2-7B experiments. This supports using the local OpenReview supplement as the primary artifact and using Table 1/Figure 3 as the reproduction target.

LoRA+ is the strongest external anchor for the separate-learning-rate proposal direction. The arXiv abstract says ordinary LoRA is suboptimal for large-width models because A and B use the same learning rate, and that LoRA+ improves performance and speed at the same compute cost. This makes asymmetric A/B learning rates a legitimate research direction rather than a random sweep.

Recent 2025-2026 federated LoRA papers crowd several naive novelty claims. ADF-LoRA and TAD-LoRA extend alternating LoRA ideas to decentralized federated learning and discuss phase-state mismatch, block-wise divergence, topology-induced cross terms, and LoRA-factor misalignment. RD-LoRA builds on alternating freezing and adds routing/decomposition and adaptive aggregation for heterogeneous settings. FedRot-LoRA argues rotational misalignment is a major source of factor-averaging error and proposes orthogonal alignment. SDFLoRA targets rank/data heterogeneity and privacy-aware decoupling. LA-LoRA targets differential privacy and local alternation, reporting large gains over RoLoRA in strict DP settings. FedMomentum targets mathematically correct aggregation plus SVD reconstruction to preserve LoRA training momentum.

Strategic implication: do not claim broad novelty for “federated LoRA with better aggregation” or “alternating LoRA under heterogeneity.” Those areas are now crowded. For this course project, the safest 12/10/paper-track claim is narrower: characterize RoLoRA’s phase-specific A/B bottlenecks under the exact centralized full-participation setup of the original paper, then test proposal-compatible interventions that preserve RoLoRA’s alternation and communication profile. If the team wants a workshop-paper extension beyond the submitted proposal, partial participation remains interesting but must be framed carefully against ADF-LoRA/TAD-LoRA/RD-LoRA rather than as an unexplored “alternating LoRA under realistic FL” area.

Primary online sources used for this snapshot:

- OpenReview RoLoRA page: https://openreview.net/forum?id=e8DrPuJekZ
- LoRA+ arXiv page: https://arxiv.org/abs/2402.12354
- ADF-LoRA arXiv page: https://arxiv.org/abs/2511.18291
- TAD-LoRA arXiv page: https://arxiv.org/abs/2602.00451
- RD-LoRA OpenReview page: https://openreview.net/forum?id=6xB2mKOGqx
- FedRot-LoRA arXiv page: https://arxiv.org/abs/2602.23638
- LA-LoRA arXiv page: https://arxiv.org/abs/2602.19926
- SDFLoRA arXiv page: https://arxiv.org/abs/2601.11219
- FedMomentum arXiv page: https://arxiv.org/abs/2603.08014
