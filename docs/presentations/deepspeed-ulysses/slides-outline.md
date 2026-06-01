# Group 7 — DeepSpeed Ulysses presentation outline

Goal: make the mechanism understandable from the visuals before the speaker explains it. Presentation slot: Tuesday 9 June 2026, 14:30, Group 7.

The deck is now a **9-slide / 18-state visual explainer**. The right arrow advances animation states first, then slides.

## Story arc

1. **Context wall** — 2 states
   A short token strip fits in one GPU; then it grows past the GPU memory boundary.
   Takeaway: one long document/genome/timeline can be a single training example, making activation memory the bottleneck.

2. **Missing axis** — 2 states
   Existing parallelism covers batch `B`, hidden `H`, and layers `L`; then sequence `N` is highlighted.
   Takeaway: Ulysses targets sequence length as the parallel dimension.

3. **Sequence-sharded layout** — 1 state
   Four GPUs each own four toy tokens, with all heads still conceptually present.
   Takeaway: before attention, memory is saved by splitting tokens across devices.

4. **All-to-all layout swap** — 3 states
   Before: GPUs own token chunks. Moving: packets cross through a peer all-to-all collective. After: GPUs own all tokens for a subset of heads.
   Takeaway: all-to-all is the central layout transform, not decorative communication.

5. **Local attention** — 2 states
   Zoom into one GPU; then the attention matrix lights up.
   Takeaway: after the swap, ordinary attention can run locally per head group.

6. **Transformer block recipe** — 2 states
   Q/K/V → all-to-all → attention → all-to-all back, with the communication steps highlighted.
   Takeaway: the Transformer math stays the same; activation placement changes twice.

7. **Scaling intuition** — 2 states
   Show `N = 1M`, `P = 64`, `N/P ≈ 16K`; then compare baseline and Ulysses communication bars.
   Takeaway: per-GPU token slice and communication shrink with sequence parallel degree.

8. **Claims + critique** — 2 states
   Show headline paper claims; then reveal assumptions.
   Takeaway: strong reported results, but fast all-to-all, head divisibility, and local attention kernels matter.

9. **Final takeaway** — 2 states
   Repeat the mechanism as a four-part map: tokens → heads → attention → tokens; then reveal Q&A prompts.
   Takeaway: Ulysses makes sequence length a parallel dimension by temporarily changing the activation layout.

## Rehearsal spine

Repeat this phrase several times:

> **tokens → heads → attention → tokens**

That is the mechanism the audience should remember.
