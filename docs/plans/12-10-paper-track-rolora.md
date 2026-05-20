# Push the RoLoRA final project from solid reproduction to 12/10 paper-track work

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

## Purpose / Big Picture

The goal is to turn the current CS4725 RoLoRA project from a well-organized reproduction attempt into a top-grade, professor-interesting research project. A merely good project reproduces some RoLoRA results and tries one improvement. A 12/10 project makes a clear scientific claim, proves the claim with a credible reproduction baseline, runs controlled ablations, explains failures honestly, and packages the work so the course staff can see workshop-paper potential.

The observable outcome is a week-9 final report and presentation where every major claim cites a command, config, seed, log, and plot; where the improvement story has one primary thesis rather than three disconnected tweaks; and where the team can show a clean artifact trail from the original RoLoRA paper to local sanity checks, paper-scale reproduction cells, and improvement experiments.

## Progress

- [x] (2026-05-20 18:55 Europe/Amsterdam) Reviewed local paper, proposal, lecture rubric, repo README, kickoff TODOs, and experiment ledger.
- [x] (2026-05-20 18:55 Europe/Amsterdam) Confirmed current repo has local evidence only, not paper-comparable RoBERTa-Large reproduction yet.
- [x] (2026-05-20 18:55 Europe/Amsterdam) Created this 12/10 paper-track plan under `.plans/12-10-paper-track-rolora.md`.
- [ ] Assign named owners for infrastructure/baselines, algorithm/ablations, and improvement/analysis.
- [ ] Create a single live progress board in `docs/progress.md` or `omx_wiki/current-state.md`.
- [ ] Run and log `make table1-medium-all`, then add results to `experiments/ledger/README.md`.
- [ ] Add a RoBERTa-Large one-round feasibility config and run it locally or on the first available GPU.
- [ ] Decide the primary improvement thesis for the final report and record it as an ADR.
- [ ] Start the report skeleton in `report/` before new results arrive.

## Surprises & Discoveries

- Observation: The current repo is stronger on engineering hygiene than on research-control hygiene.
  Evidence: `README.md` defines layout, commands, and status; `docs/decisions/` has ADRs; `experiments/ledger/README.md` records evidence. But `docs/kickoff.md` still has TODOs for owner mapping, tech lead, cadence, Git workflow, and paper-presentation roles.

- Observation: The proposal commits to three improvement directions, while the deep-research plan’s strongest paper-shaped angle is partial participation. This is a strategic tension.
  Evidence: `README.md` and `docs/research/project-proposal.pdf` commit to improved initialization, separate A/B learning rates, and adaptive server-side optimization. `docs/research/deep-research-plan.md` recommends partial participation as the strongest paper niche, but project guidance says that recommendation is superseded by the submitted proposal.

- Observation: The current local Table-1-shaped pilot is not a scientific result yet.
  Evidence: `experiments/ledger/README.md` explicitly says the RoBERTa-base QNLI local pilot is too tiny for paper-comparable numbers, and `README.md` says full Table 1 requires RoBERTa-Large across GLUE tasks/client counts/multiple seeds.

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

Current outcome: the plan identifies the critical gap between the current repository state and a top-grade final project. The repository is ready for serious execution, but the research story, ownership, report skeleton, and paper-scale experiment matrix are not yet mature enough for a 12/10 outcome.

The main lesson is that the project should stop thinking in terms of “run more experiments” and start thinking in terms of “prove one compelling claim with the minimum experiment set that withstands scrutiny.”

## Context and Orientation

This repository is a TU Delft CS4725 project reproducing and extending RoLoRA: “Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA.” RoLoRA is a federated learning method for LoRA adapters. LoRA represents a weight update as the product of two low-rank matrices, commonly called A and B. In federated learning, several clients train locally and a central server aggregates updates. Directly averaging A and B separately does not equal averaging the full product A times B, so ordinary FedAvg over LoRA factors introduces an aggregation error. RoLoRA avoids this by alternating: in one communication round A is frozen and B is trained/aggregated; in the next round B is frozen and A is trained/aggregated. Because one factor is shared and fixed in each round, aggregation is exact for the active factor.

The local source documents are the authority. `docs/research/paper-rolora.pdf` is the paper. `docs/research/project-proposal.pdf` is the submitted proposal. `docs/research/lecture-01-introduction.pdf` gives assessment: paper presentation is 20%, research project is 60%, homeworks are 20%; project deliverables are final report and final presentation in week 9, with draft report in week 8. `README.md` summarizes this in repository form. `experiments/ledger/README.md` records evidence already collected.

The current repo has good engineering structure: `Makefile` commands, `scripts/` runners, `experiments/configs/` YAML configs, `tests/` invariants, `docs/decisions/` ADRs, and local results logs. The current repo does not yet have strong research-control structure: there is no single progress board, no explicit full experiment matrix with owners and statuses, no report skeleton, and no named ownership split.

## Critical Assessment Against a 12/10 Standard

The current project is good for week 3, but it is not yet a 12/10 project. A 12/10 project must make the grader think: “this team understood the paper deeply, reproduced the central claim credibly, discovered something non-obvious, and packaged it like a small research artifact.” The current state mostly proves that the team can run and patch the harness. That is necessary, but not sufficient.

The biggest weakness is that the improvement plan is currently too diffuse. “Improved initialization,” “separate learning rates,” and “adaptive server-side optimization” are each plausible, but if reported as three independent tweaks they will read like a parameter sweep rather than a contribution. The fix is to unify them around A/B phase dynamics. The report should ask: “RoLoRA’s exact aggregation relies on alternating A and B, but are the two phases equally stable and equally optimized?” Then each improvement becomes part of the same investigation.

The second weakness is that the reproduction target is still underspecified operationally. The proposal says Table 1 and Figure 3. The repo says full Table 1 is too expensive locally. The team needs a graded fallback ladder. Minimum acceptable reproduction for a strong grade is one GLUE dataset at RoBERTa-Large with 3, 20, and 50 clients across LoRA, FFA-LoRA, and RoLoRA, plus convergence curves for the 50-client case. Better is three datasets. Full 5-task Table 1 is ideal but not required if it prevents deeper analysis.

The third weakness is report risk. The report directory is empty. If writing starts in week 8, the project will look like an experiment dump. A top-grade report needs a narrative skeleton now, with placeholders for figures and tables. Every experiment should be run because it fills a named slot in the report, not because it is “next.”

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

Milestone 1 is to finish project-control structure. Add `docs/progress.md` with a table of workstreams, owners, status, blockers, next action, and evidence path. Add `docs/experiment-matrix.md` with the exact reproduction and improvement matrix. Add a new ADR under `docs/decisions/` that ratifies the unified improvement thesis. This should happen before more large experiments, because otherwise new evidence will be hard to organize.

Milestone 2 is to finish the local-to-large bridge. Run `make table1-medium-all` and summarize it. Then add a RoBERTa-Large one-round feasibility config derived from the supplement harness. The acceptance criterion is not accuracy; it is that the model loads, data path works, memory use is known, and the log contains the same method markers as local pilots. This de-risks the first real cluster job.

Milestone 3 is the minimum credible reproduction. Run one RoBERTa-Large dataset, preferably MNLI if feasible and QNLI if time/compute is tight, for 3, 20, and 50 clients across LoRA, FFA-LoRA, and RoLoRA. Use at least three seeds for the most important 50-client comparison; if compute is constrained, use one seed for 3/20 and three seeds for 50. Generate Figure-3-style convergence curves for 50 clients. The acceptance criterion is a table and curve that directly test the paper’s central claim.

Milestone 4 is the improvement characterization. Implement or expose experiment knobs for the three committed directions. Improved initialization should include standard PEFT init, orthogonal A, and one SVD/PCA-like data-informed A if feasible. Separate learning rates should sweep ratios such as A:B equals 1:1, 1:4, 1:8, 1:16, and 4:1 depending on which phase appears undertrained. Adaptive server-side optimization should start with FedAvg versus server momentum or server Adam on the active factor only. The acceptance criterion is not that all three beat RoLoRA; it is that each answers a precise question about phase-specific dynamics.

Milestone 5 is report and presentation hardening. Create the report skeleton now. Each figure should have an owner, expected command, and fallback. The paper presentation should focus on RoLoRA itself, but it should preview the project extension in the discussion section. The final project presentation should be story-first: problem, reproduction trust, phase-specific diagnosis, intervention results, limits.

## Concrete Steps

From repository root `/Users/vliftode/personal/scalable-learning`, first verify the baseline:

    uv run pytest -q
    uv run ruff check .

Expect all tests to pass. On 2026-05-20 this was observed as 23 tests passing and Ruff reporting all checks passed.

Create the control docs:

    touch docs/progress.md docs/experiment-matrix.md

`docs/progress.md` should have rows for infrastructure/baselines, algorithm/ablations, improvement/analysis, report/presentation, and cluster/access. Each row must have owner, this-week target, next command, evidence path, and blocker.

`docs/experiment-matrix.md` should define the minimum reproduction matrix and the stretch matrix. The minimum matrix should be one dataset, three methods, three client counts, rank 4, and enough seeds to show variance at 50 clients. The stretch matrix should add more GLUE tasks and rank-8/rank-2 ablations only after the minimum matrix is underway.

Run the next local rung:

    make table1-medium-all
    make table1-medium-summary

Add the outputs to `experiments/ledger/README.md` with date, command, scale, evidence path, and interpretation. If the three local methods have identical accuracy, explicitly write that this is expected or not interpretable at tiny scale; do not overclaim.

Add the report skeleton under `report/`. The first committed version can be markdown or LaTeX, but it must include figure placeholders. Minimum placeholders are: RoLoRA mechanism diagram, MNIST sanity result, reproduction table, 50-client convergence curve, phase-specific improvement ablation, limitations table.

Before cluster runs, update `docs/setup/delftblue.md` and `slurm/README.md` with TA-confirmed instructions. Do not rely on provisional partition names as final truth.

## Validation and Acceptance

The project is on a 12/10 trajectory only when the following are true.

First, `make check` passes before cluster runs. Second, every experiment claim has a row in `experiments/ledger/README.md`. Third, every major report figure has a config and log path. Fourth, the team can answer: “What is the one scientific claim beyond reproduction?” in one sentence. Fifth, the midterm review can show a live table of done/running/blocked experiments rather than a verbal status update.

The final 12/10 acceptance bar is: the report contains a credible reproduction of RoLoRA’s client-scaling claim, at least one improvement or characterization result that teaches something about A/B alternation, clear negative-result handling, and a reproducibility appendix. The presentation must lead with the problem and contribution, not implementation details.

## Idempotence and Recovery

All documentation steps are safe to repeat. If an experiment fails, do not delete the log. Add it to the ledger as a failed run with the error and next action. If cluster access is delayed, shift to the strongest local substitute: RoBERTa-base longer runs, one RoBERTa-Large feasibility probe on any available GPU, and deeper MNIST/linear characterization of phase-specific behavior. If the supplement becomes a blocker, switch to the FedSA-LoRA fallback per ADR 0001 and document the switch in a new ADR.

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
