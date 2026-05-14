# Paper presentation outline — RoLoRA (W7–8)

Target slot: **10–12 minutes presentation + 5–6 minutes Q&A** (15–18 min total).

Rubric (from CS 4725 lecture-1):
- **Motivation / problem statement — 20%**
- **Proposed solution / contribution — 40%**
- **Evaluation — 25%**
- **Leading in-class discussion — 15%**

Plan the slide count by weight: about 12 content slides + a title + a discussion-prompts slide. Allocate ≈ 2 minutes per content section.

---

## 1. Motivation (20%, ~2–3 slides, ~2 min)
- **Slide 1 — Why federated + LoRA at all?**
  Privacy-sensitive data lives at the edge (hospitals, phones, banks). Full fine-tuning is bandwidth-prohibitive. LoRA cuts comm to ~MB-per-client per round.
- **Slide 2 — The math bug.**
  Show the identity that fails: `avg(A_i · B_i) ≠ avg(A_i) · avg(B_i)`. Concrete tiny-tensor example.
- **Slide 3 — Why prior fixes are unsatisfying.**
  FlexLoRA / FLoRA: aggregate the products (defeats the comm savings). FFA-LoRA: freeze A (kills expressiveness, especially at low rank and many clients). Set up the question: can we get exact math without sacrificing expressiveness?

## 2. Solution (40%, ~5–6 slides, ~4–5 min)
- **Slide 4 — The alternation idea.**
  Diagram: odd round → A frozen + shared, train B, average B (exact). Even round → swap. Same per-round comm as FFA-LoRA; both factors learned.
- **Slide 5 — Why this is exact, in words.**
  Because each round has a shared frozen factor across clients, the sum of products factors out: `avg(A_const · B_i) = A_const · avg(B_i)`. No aggregation error.
- **Slide 6 — The toy: rank-1 linear regression.**
  State the model and what the paper proves. Punchline: RoLoRA has exponential angle-convergence; FFA-LoRA's loss is lower-bounded by `sin²(initial angle)`.
- **Slide 7 — Non-convex convergence.**
  RoLoRA matches FedAvg's `O(1/√T)` stationary-point guarantee. No worse than the standard FL theory.
- **Slide 8 — What's not solved.**
  Paper explicitly defers partial participation; says little about heterogeneous ranks across clients; assumes full participation; theory is rank-1 toy. Tee up our improvement directions later.

## 3. Evaluation (25%, ~3 slides, ~2–3 min)
- **Slide 9 — Figure 2 (MNIST 2-layer MLP).**
  Cleanest validation: FFA-LoRA plateaus near 55%, RoLoRA keeps climbing. We reproduced this on a laptop (gestures to our results).
- **Slide 10 — Table 1 (RoBERTa-Large, 50-client cliff).**
  LoRA collapses to ~52%, FFA-LoRA holds ~70%, RoLoRA holds ~83%. The headline.
- **Slide 11 — Robustness.**
  More clients, fewer trainable parameters, more local steps — RoLoRA holds across all three axes. Note the local-steps ablation explicitly.

## 4. Discussion-leading (15%, 1–2 slides + Q&A bank)
- **Slide 12 — Provocations.**
  Three prompts to seed discussion:
  1. The alternation is parameter-equivalent to FFA-LoRA per round; why is it not equivalent in the limit?
  2. The theory rests on a rank-1 linear model. How worried should we be that the empirical gap is mostly explained by something orthogonal — e.g., learning-rate decoupling between A and B (LoRA+)?
  3. Real FL has stragglers. The paper assumes full participation. What goes wrong under partial participation? (Set up our extension.)
- **Slide 13 — Our extension teaser.**
  Three directions we're trying, in 30 seconds: orthogonal/SVD init for A; LoRA+-style separate LRs; adaptive server-side optimisation.
- **Q&A bank (not slides):**
  - "What happens if you use rank > 4?"
  - "Why not always train both factors and just project?" → FlexLoRA reply.
  - "Does this generalise beyond GLUE / commonsense reasoning?" → Llama-2-7B numbers.
  - "How does this interact with DP?" → Table 20 result, but only one setting.

## Production checklist
- One person leads slides; another handles Q&A.
- Practice the timing — under-prepared 15-min talks always overrun.
- Slides in `report/presentation/` (LaTeX-Beamer or Keynote — TBD at kickoff).
