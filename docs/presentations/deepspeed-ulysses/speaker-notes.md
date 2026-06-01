# Speaker notes — DeepSpeed Ulysses

## Opening framing

This is a systems paper. The key point is not a new model architecture and not an approximate-attention trick. The key point is a data-layout transform that makes **sequence length** parallelizable.

## The one sentence to repeat

> Ulysses splits the sequence across GPUs for most of the layer, uses all-to-all to give each GPU the full sequence for a subset of attention heads, computes ordinary local attention, then all-to-alls back.

## Slide-by-slide notes

1. **Title:** Say the problem: long-context training hits memory and communication walls.
2. **Motivation:** Use examples; do not over-explain each domain.
3. **Gap:** Name the four axes. Emphasize sequence length is the unsolved axis.
4. **Core idea:** Make the audience understand the layout switch before any formulas.
5. **Mental model:** During attention each GPU has full sequence context for a subset of heads.
6. **Mechanism:** Two all-to-alls around attention; everything else stays sequence-parallel.
7. **Scaling:** Formula slide. Carefully explain the assumption: N and P scale together.
8. **Microbenchmark:** Same data moved, much lower time. This is about contention/layout, not magic compression.
9. **Practical system:** Why this is usable: ZeRO-3 + existing attention kernels + DeepSpeed integration.
10. **Results:** Present headline numbers quickly.
11. **Memory:** ZeRO handles parameters; Ulysses handles long-sequence activations.
12. **ClimaX:** Nice concrete use case; use if time allows.
13. **Critique:** Strongest discussion slide. Mention network dependence and head-count cap.
14. **Takeaway:** Repeat: layout transform, not approximate attention.
15. **Discussion:** Ask one question yourself if the room is quiet.

## Likely Q&A

**Q: Is this the same as FlashAttention?**
No. FlashAttention is a local attention kernel optimization. Ulysses is distributed sequence parallelism. The paper claims it can use FlashAttention inside each GPU after the all-to-all layout transform.

**Q: Why does communication become constant?**
Per GPU, communication is proportional to `N/P`. If the sequence length `N` grows in proportion to device count `P`, then `N/P` is constant.

**Q: What is the main limitation?**
All-to-all depends on fast interconnects, and sequence parallel degree is limited by attention heads.

**Q: What should we compare it to today?**
Ring Attention / context parallelism are natural follow-ups; the question is whether they replace Ulysses or compose with it.
