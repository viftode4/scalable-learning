# DeepSpeed Ulysses presentation outline

Target: 10-12 minutes + 5-6 minutes Q&A.

Deck: [`deepspeed-ulysses-presentation.html`](deepspeed-ulysses-presentation.html)

## Story arc

1. **Title — Training past the context wall**
   Establish the paper and the central systems problem.
2. **Why long sequences matter**
   Generative AI, genomics, climate, healthcare.
3. **The gap**
   Batch / hidden / layers are covered by existing parallelism; sequence length is not.
4. **Core idea**
   Split sequence across GPUs, then all-to-all into head-parallel attention.
5. **Mental model**
   What each GPU sees during attention: full sequence, subset of heads.
6. **Mechanism**
   Two all-to-alls per block around local attention.
7. **Why it scales**
   Communication comparison: `4Nd` vs `4Nd/P`.
8. **Microbenchmark**
   34 ms gather/scatter vs 4.9 ms all-to-all.
9. **Practical system**
   ZeRO-3, attention-agnostic design, minimal adoption friction.
10. **Headline results**
   1M+ tokens, 2.5× throughput, 4× length, 10× less communication.
11. **Memory result**
   ZeRO partitions weights; Ulysses partitions sequence activations.
12. **ClimaX case study**
   4× longer sequences on same hardware.
13. **Critique**
   Network dependence, head-count cap, fixed-GPU-budget caveat, newer context-parallel methods.
14. **Takeaway**
   Sequence length becomes a first-class parallel dimension.
15. **Discussion**
   Use questions to lead the room.

## Suggested speaker split

- **Presenter 1:** slides 1-3 — motivation and gap.
- **Presenter 2:** slides 4-9 — Ulysses mechanism and system design.
- **Presenter 3:** slides 10-15 — results, critique, and discussion.

## Rehearsal rule

If the deck runs long, cut slide 11 or 12 first. Do **not** cut the mechanism or critique slides.
