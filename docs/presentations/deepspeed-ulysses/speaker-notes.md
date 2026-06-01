# Speaker notes — Group 7 DeepSpeed Ulysses

Presentation slot: Tuesday 9 June 2026, 14:30, Group 7. Target: 9-10 minutes + 4-5 minute Q&A.

One-sentence version:

> Ulysses splits the sequence across GPUs for most of the layer, uses all-to-all to regroup activations so each GPU has the full sequence for a subset of attention heads, computes normal local attention, then all-to-alls back.

## Slide-by-slide script

### 1. Context wall

Step 1: “A short prompt fits in one GPU.”
Step 2: “But the paper is about extreme long sequences: books, genomes, climate fields, or long patient histories. At these lengths, activation memory becomes the bottleneck.”

Do not spend long here. The point is only to motivate why sequence length is the bottleneck.

### 2. Missing axis

“We already know how to split work across batch, hidden dimension, and layers. The missing axis is `N`, sequence length. Ulysses asks: can we make sequence length itself a parallel dimension?”

### 3. Sequence-sharded layout

“Before attention, Ulysses stores different token ranges on different GPUs. In the toy example, GPU 0 has tokens 0–3, GPU 1 has tokens 4–7, and so on. This is memory-friendly, but attention needs global context.”

### 4. All-to-all layout swap

This is the most important slide. Walk slowly.

Step 1: “Before the swap, each GPU owns a token chunk.”
Step 2: “All-to-all is a peer collective: every GPU exchanges pieces with the others, not through a central router.”
Step 3: “After the swap, each GPU has all token colors — our shorthand for the full sequence — but only for a subset of heads.”

Key phrase:

> “Same tokens, different layout.”

### 5. Local attention

“Now GPU 0 can compute attention for heads 0–1 because it has the full sequence for that head group. This is why the all-to-all was worth doing. The attention formula is not changed; the tensors were moved so the formula can run locally.”

### 6. Transformer block recipe

“Inside each Transformer block the rhythm is: project Q/K/V, all-to-all tokens into heads, compute attention, all-to-all back into tokens. Then feed-forward and the rest of the layer can continue in sequence-parallel layout.”

Memorable phrase:

> tokens → heads → attention → tokens

### 7. Scaling intuition

“If the full sequence has `N` tokens and we split across `P` GPUs, each GPU handles about `N/P` tokens in the sequence-parallel parts. This is why the paper argues per-GPU communication becomes manageable when sequence length and GPU count scale together.”

Use the bars: the paper states Megatron-style sequence parallelism has per-link communication `4Nh`, while DeepSpeed sequence parallelism has `4Nh/P`, i.e. `O(N/P)`.

### 8. Claims + critique

Claims:
- 1M+ token context on a 1.2B GPT-style model.
- Up to 2.5× throughput versus the existing SOTA baseline.
- 4× larger sequence length than existing systems, including over one million tokens.
- Over 10× communication-volume reduction.

Critique:
- All-to-all needs a fast interconnect.
- Sequence-parallel size must divide the number of attention heads in current implementation guidance.
- The method is most compelling when sequence length is truly the bottleneck.

End with:

> “The paper’s contribution is a systems layout trick: make sequence length parallelizable by temporarily turning token shards into head shards.”

### 9. Final takeaway

Step 1: Repeat the map slowly: “tokens → heads → attention → tokens.”
Step 2: Use the three prompts as Q&A openings if the class is quiet.

Closing sentence:

> “DeepSpeed Ulysses is powerful because it turns sequence length from a memory wall into a distributed layout problem.”

## Likely Q&A

**Is this FlashAttention?**
No. FlashAttention is a local attention kernel. Ulysses is distributed sequence parallelism and can use efficient local attention kernels inside each GPU.

**Why is the number of heads important?**
During attention, GPUs are assigned head groups. Implementation docs state the number of attention heads must be divisible by the sequence-parallel size; intuitively, Ulysses assigns non-overlapping head groups to devices.

**What is the main limitation?**
The all-to-all exchange depends heavily on fast GPU interconnect. On weak networks, the communication can dominate.
