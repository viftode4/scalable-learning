# DeepSpeed Ulysses presentation outline

Target: **8-10 minutes** plus Q&A. Keep the slides visual; put detail in speech.

Deck: [`deepspeed-ulysses-presentation.html`](deepspeed-ulysses-presentation.html)

## Final short story arc

1. **Title — Training past the context wall**
   Frame Ulysses as a systems paper about parallelizing sequence length.
2. **Examples**
   Book/legal case, genome string, climate grid, patient timeline.
3. **The gap**
   Batch, hidden, and layer axes are covered; sequence length is the missing axis.
4. **Core idea**
   Change layout, not attention math: sequence-parallel outside attention, head-parallel during attention.
5. **Toy example**
   16 tokens / 4 GPUs / 8 heads. This is the main explanation slide.
6. **Mechanism**
   Q/K/V → all-to-all → local attention → all-to-all back.
7. **Scaling**
   Ground `4Nd/P` with `N = 1M`, `P = 64`, `N/P ≈ 16K`.
8. **Results**
   1M+ tokens, 2.5× throughput, 4× longer sequences, 10× less communication.
9. **Critique**
   Network dependence, head-count cap, N/P scaling assumption, newer context-parallel methods.
10. **Takeaway + Q&A**
   Tokens → heads → attention → tokens, plus three discussion prompts.

## Suggested speaker split

- **Presenter 1:** slides 1-3 — examples and problem.
- **Presenter 2:** slides 4-7 — mechanism and scaling.
- **Presenter 3:** slides 8-10 — results, critique, and discussion.

## Rehearsal rule

Do not rush slide 5. If time is short, compress slide 8; do not cut the toy example or critique.
