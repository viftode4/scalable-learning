# CS4725 Group Project — Grading Rubric (official)

Source: course staff ("Group Project Rubric (New)"). This is the authoritative
rubric for the graded **research project** (60% of the course). Save/keep this
current — it supersedes any improvised rubric.

## Rubric (60 points total)

| Item | Points | Covered by |
|---|---|---|
| **Quality of the report** — writing quality, completeness, correctness | **15 / 60** | report |
| **Reproduction Results** — alignment with paper results and explanations | **10 / 60** | report |
| **Improvements** — significance, implementation, evaluation, novelty | **15 / 60** | report |
| **Quality of the presentation** — clarity, smoothness, focus on key results | **10 / 60** | presentation |
| **Answers to questions** — correctness, clarity, efficiency, level of expertise | **10 / 60** | live Q&A |

**Hard requirement (both deliverables):**
- **Contributions of each group member must be stated** — in **both** the
  project report *and* the presentation.

> The written report directly drives **40 of 60 points** (report quality 15 +
> reproduction 10 + improvements 15). Presentation (10) and Q&A (10) are the
> live defense.

## Related: Paper Presentation Rubric (separate 20% component)

For the *paper presentation* (presenting an existing paper, weeks 7–8) — from
`docs/research/lecture-01-introduction.pdf` (slide 10): Motivation/problem 20% ·
Proposed solution + novelty 40% · Evaluation 25% · Leading the discussion 15%.

## How our project maps (as of 2026-06-18)

Grounded in the verified results (`runs/REGISTRY.md`) and the reviewer feedback.

| Rubric item | Pts | Where we stand | Highest-leverage action |
|---|---|---|---|
| Report quality | 15 | Strong, clean prose; honest limitations | Fix the **Table 1 error-bar inconsistency** (CI95 vs pop-std mix) and the **`5·10⁻⁵`→`5·10⁻³`** typo — these are the "correctness" sub-criterion |
| Reproduction | 10 | Solid: two-scale reproduction, honest about reductions, numbers verified against logs | Keep the "not a complete reproduction" framing; nothing major needed |
| Improvements | 15 | **Thinnest area**: modest gains, headline (orth-A+BBA, 88.32) on only 2 seeds | Sharpen the **init×schedule novelty**; fold in the **running LR sweep** (verifies SVD's limited improvement); optionally the orth-A+BBA seed-2 + default-init control |
| Presentation | 10 | n/a (live) | Lead with the strongest result; state per-member contributions on a slide |
| Q&A | 10 | n/a (live) | Be ready on: why proxy ≠ Table 1, the seed count, the error-bar convention |
| Contributions stated | gate | ✅ in report ("Team contributions" section) | Ensure it's also **on a presentation slide** |

**Takeaway:** "Improvements" (15) and "Report quality" (15) are the two heaviest
report items. Improvements is where we're weakest, so the LR sweep + a crisp
contribution/novelty framing are the best marginal spend — not more breadth.
