# Speaker notes — DeepSpeed Ulysses

## One sentence

> Ulysses splits the sequence across GPUs for most of the layer, uses all-to-all to give each GPU the full sequence for a subset of attention heads, computes ordinary local attention, then all-to-alls back.

## Slide-by-slide notes

1. **Title:** This is not a new model architecture; it is a distributed layout trick for long context.
2. **Examples:** Make the stakes concrete. Say: one sample can be a whole book, genome segment, climate field, or patient record.
3. **Gap:** Existing parallelism splits batch, hidden dimension, or layers. Long-context training needs to split the sequence itself.
4. **Core idea:** Explain the layout swap. Outside attention: token shards. During attention: head shards with full sequence.
5. **Toy example:** Walk left to right slowly. Before: GPU 0 has tokens 1-4, GPU 1 has tokens 5-8, etc. After all-to-all: each GPU has all 16 tokens but only some heads.
6. **Mechanism:** Four steps: Q/K/V, all-to-all, local attention, all-to-all back.
7. **Scaling:** Use the numbers first: with 1M tokens and 64 GPUs, each GPU carries about 16K tokens outside attention. Then show why the communication term divides by P.
8. **Results:** Keep this short. The exact numbers are less important than the pattern: longer context, faster training, less communication.
9. **Critique:** This is where we show understanding. Ulysses depends on fast all-to-all, is limited by attention heads, and should be compared to newer context-parallel/ring methods.
10. **Takeaway + Q&A:** Repeat “tokens → heads → attention → tokens”, then ask one discussion question.

## Likely Q&A

**Is this the same as FlashAttention?**
No. FlashAttention is a local attention kernel. Ulysses is distributed sequence parallelism and can use FlashAttention inside each GPU.

**Why does communication become constant?**
Per GPU, communication is proportional to `N/P`. If sequence length `N` grows with device count `P`, then each GPU's slice stays about the same size.

**Main limitation?**
All-to-all performance depends on fast interconnects, and sequence-parallel degree is capped by attention heads.
