# Kickoff agenda — open items for the team

Use this as the agenda for the first meeting. Items already resolved (grading split, paper-presentation paper, improvement directions) have been moved out of the questions and into `README.md` / `AGENTS.md`.

## Resolved (no longer open)
- **Assessment split:** 20% paper presentation (W7-8) + 60% project + 20% homeworks.
- **Paper-presentation paper:** RoLoRA itself (same as project paper).
- **Improvement directions:** three committed in the proposal — improved init / separate LRs / adaptive server-side optimization.
- **Harness:** primary = OpenReview supplement; fallback = FedSA-LoRA fork (submodule).
- **Supplement setup:** downloaded, audited, installed in isolated Python 3.9 env, smoke-tested locally.
- **Env:** uv-managed Python 3.11, `peft==0.10.0` pinned.

## Open items

### 1. Cluster access — waiting for TAs
- DelftBlue + DAIC access is TA-driven. Dennis Heijmans / Rui Wang will share instructions in a week-4 lecture or BrightSpace announcement.
- **TODO:** designate one team member to track BrightSpace + course email for the announcement. If nothing arrives by mid-W4, ping Dennis directly.
- Do **not** file a TOPdesk request without TA guidance.

### 2. Time commitment
- **TODO:** realistic hours/week per person. Honest answers — a 6h/week and 20h/week split needs different ownership.

### 3. Team split by ownership layer
Default (recommended by the deep-research plan):
- **Infrastructure & baselines** — harness, datasets, logging, plotting.
- **Algorithm & ablations** — RoLoRA / FFA-LoRA / LoRA implementations; alternation correctness; exactness asserts.
- **Improvement & analysis** — one of the three committed angles, plus the MNIST toy sanity check.

**TODO:** map names to layers. Discuss whether one person should own all three improvement angles or whether to split (e.g., one person per angle).

### 4. Tech-lead / tiebreaker
**TODO:** designate.

### 5. Paper-presentation prep (W7-8)
- 15-18 min slot on the RoLoRA paper (10-12 present + 5-6 Q&A).
- Rubric: motivation 20% / solution 40% / evaluation 25% / discussion-leading 15%.
- Outline scaffold lives in `docs/templates/paper-presentation-outline.md`.
- **TODO:** assign roles — who leads slides, who runs Q&A.

### 6. Individual homeworks (W4 and W5)
- Two written homeworks, individual deliverables, 20% of the grade.
- They eat hours that would otherwise go to the project.
- **TODO:** flag on calendar; coordinate so reproduction sprints don't collide with homework deadlines.

### 7. Communication + cadence
Default: 2× weekly, 30 minutes. Once early in the week to plan, once mid-week to unblock.
**TODO:** channel (Discord / Slack / WhatsApp / Telegram)? Meeting time?

### 8. Git workflow
Default: branches per person, PRs reviewed by ≥1 other before merge to `main`.
**TODO:** confirm.

### 9. Reproduction milestones (per deep-research roadmap)
- W4 setup + literature review (parallel).
- W5 reproduction runs start (LoRA baseline on 3-client MNLI within ±2% of paper).
- W6 reproduction at scale + improvement work begins.
- W7-8 improvement experiments + report writing.
- W9 finalization.

**TODO:** sanity-check this against actual hours-budget once item #3 is settled.
