# Kickoff agenda — open questions for the team

Use this as the agenda for the first meeting. Each question has a default suggested by the deep-research plan; the team needs to confirm or override.

## 1. Compute access
- DelftBlue and DAIC are confirmed available. Local machines for smaller iteration.
- **TODO:** Who requests the faculty share on DelftBlue via TOPdesk, and by when? (Without it, queue times on `gpu-a100-small` may be 8–24h on busy days.)

## 2. Time commitment
- **TODO:** Realistic hours/week per person. Honest answers — a 6h/week and 20h/week split needs different ownership.

## 3. Team split — by ownership layer
Default (recommended by the deep-research plan):
- **Infrastructure & baselines** — owns the harness, datasets, logging, plotting.
- **Algorithm & ablations** — owns RoLoRA / FFA-LoRA / LoRA implementations, alternation correctness, exactness asserts.
- **Improvement & analysis** — owns the improvement angle and the MNIST toy (Fig. 2) sanity check.

**TODO:** Map names to layers.

## 4. Tech-lead / tiebreaker
Not a boss — just a deciding vote when there's a technical disagreement.
**TODO:** Designate.

## 5. Grading awareness
- **TODO:** Confirm the grading weight split (reproduction vs. improvement vs. writeup) with the course coordinator. Affects time allocation.

## 6. Improvement angle ratification
Deep-research recommendation (primary): **Robust RoLoRA under partial client participation + communication-time-aware scheduling.**

- Hypothesis: stale A (projection) hurts more than stale B (head).
- Paper explicitly defers partial participation in Appendix A2.
- Has not been claimed by the Nov 2025 / Feb 2026 follow-ups (ADF-LoRA, FedRot-LoRA, FedMomentum, LA-LoRA, SDFLoRA).

Secondary experiments that fit the same codebase: LoRA+ asymmetric LR sweep (Fig. 6 in the paper only tested 2× / 4×; LoRA+ recommends ~16×) and warmup-then-freeze (Fig. 5 mixing ablation didn't try short warmup).

**TODO:** Confirm angle, or substitute.

## 7. Communication + cadence
Default: 2× weekly, 30 minutes. Once early in the week to plan, once mid-week to unblock.
**TODO:** Channel (Discord / Slack / WhatsApp / Telegram)? Meeting time?

## 8. Git workflow
Default: branches per person, PRs reviewed by ≥1 other before merge to `main`. Shared running doc (Notion or just `docs/`) for experiment log.
**TODO:** Confirm.

## 9. Code-availability action items (do these in week 1, parallelized)
- One person downloads the OpenReview supplementary zip from `u4mobiHTJl` (the authoritative RoLoRA implementation).
- One person emails the first authors (`shuangyi.chen@mail.utoronto.ca`, `yuanxin.guo@mail.utoronto.ca`) asking for a public/cleaned-up code release.
- One person forks `Pengxin-Guo/FedSA-LoRA` as the harness baseline. (See `code/harness/README.md` for the intended layout.)
