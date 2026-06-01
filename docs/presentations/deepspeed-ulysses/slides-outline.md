# DeepSpeed Ulysses presentation outline

Target: 10-12 minutes + 5-6 minutes Q&A.

Deck: [`deepspeed-ulysses-presentation.html`](deepspeed-ulysses-presentation.html)

## Story arc

1. **Title — Training past the context wall**
   Establish the paper and the central systems problem.
2. **Why it matters**
   Real inputs are naturally long: documents, genomes, climate grids, patient records.
3. **The gap**
   Batch / hidden / layers are covered by existing parallelism; sequence length is not.
4. **Core idea**
   Split sequence across GPUs, then all-to-all into head-parallel attention.
5. **Toy example**
   16 tokens / 4 GPUs / 8 heads: show layout transform concretely.
6. **Mental model**
   What each GPU sees during attention: full sequence, subset of heads.
7. **Mechanism**
   Two all-to-alls per block around local attention.
8. **Why it scales**
   Concrete N/P intuition plus `4Nd` vs `4Nd/P`.
9. **Microbenchmark**
   34 ms gather/scatter vs 4.9 ms all-to-all.
10. **Practical system**
   ZeRO-3, attention-agnostic design, minimal adoption friction.
11. **Headline results**
   1M+ tokens, 2.5× throughput, 4× length, 10× less communication.
12. **Memory result**
   ZeRO partitions weights; Ulysses partitions sequence activations.
13. **ClimaX case study**
   4× longer sequences on same hardware.
14. **Critique**
   Network dependence, head-count cap, fixed-GPU-budget caveat, newer context-parallel methods.
15. **Takeaway**
   Sequence length becomes a first-class parallel dimension.
16. **Discussion**
   Use questions to lead the room.

## Suggested speaker split

- **Presenter 1:** slides 1-3 — motivation and gap.
- **Presenter 2:** slides 4-10 — Ulysses mechanism and system design.
- **Presenter 3:** slides 11-16 — results, critique, and discussion.

## Rehearsal rule

If the deck runs long, cut slide 12 or 13 first. Do **not** cut the toy example, mechanism, communication formula, or critique slides.
