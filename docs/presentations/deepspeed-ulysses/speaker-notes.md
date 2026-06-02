# Speaker notes — Group 7 DeepSpeed Ulysses

Presentation slot: Tuesday 9 June 2026, 14:30, Group 7. Target: 9 minutes core + 4-5 minute Q&A.

One-sentence version:

> Ulysses shards the token sequence across GPUs, uses all-to-all collectives to regroup activations so each GPU holds all tokens for a subset of attention heads, computes ordinary local attention, then all-to-alls back. The mantra: tokens → heads → attention → tokens.

---

## Slide-by-slide script

### 1. Context wall (2 states — ~40 seconds)

**State 1:** "Start with a short prompt in one GPU. Easy."

**State 2:** "But the paper targets extreme long sequences: books, genomes, climate data, long patient timelines. At scale—call it millions of tokens—the activation memory footprint overflows one GPU. That's the context wall."

**Key point:** Do not dwell. This is motivation only. We need sequence length as a parallel axis because no single GPU can hold all activations.

**Timing:** 0:00–0:40

---

### 2. Missing axis (2 states — ~45 seconds)

**State 1:** "We already parallelize across batch size B, hidden dimension H, and layers L. Three dimensions. But token sequence length N sits unused."

**State 2:** "Ulysses asks: can we make N itself a parallel dimension? Can we split the token stream across GPUs and still run attention?"

**Key point:** Frame this as the central research question. It motivates the entire mechanism.

**Timing:** 0:40–1:25

---

### 3. Sequence-sharded layout (2 states — ~50 seconds)

**State 1:** "Here's the naive answer: shard the full token sequence across P GPUs. GPU 0 owns tokens 0 to N/P−1, GPU 1 owns tokens N/P to 2N/P−1, and so on. Each GPU stores shape [N/P, d] where d is the hidden dimension. All attention heads are still _conceptually_ present—we haven't split heads yet."

**State 2:** "This layout is memory-efficient: each GPU holds only 1/P of the full activation memory. The identity colors here represent different token chunks. But now we have a problem: attention is a global operation. Each token must attend to all other tokens in the sequence. From a single GPU's perspective, most of the key-value context is on _other_ machines."

**Key point:** Establish the identity colors (visual marker for token ranges). End with the problem: how do we give each GPU the global context it needs for attention?

**Timing:** 1:25–2:15

---

### 4. All-to-all #1: tokens → heads (3 states — ~1:20)

**This is the hero slide. Walk slowly.**

**State 1:** "Before the collective, GPU 0 owns tokens {0, 1}, GPU 1 owns tokens {2, 3}, and so on. But _both_ tokens on GPU 0 carry all 2 attention head dimensions."

**State 2:** "All-to-all is a peer-to-peer collective. Every GPU sends pieces of its tensors to every other GPU, and receives pieces back. It's not a broadcast; it's a global exchange."

**State 3:** "After the collective exchange: GPU 0 now holds _all_ 4 tokens, but only for head 0. GPU 1 holds all 4 tokens, but only for head 1. We swapped the axis. Tokens are now dense; head dimension is sharded. The transformation happens in-place via a single MPI call."

**One-liner to anchor understanding:**

> "Same tokens, different layout."

**Key point:** The all-to-all is not overhead—it's the core computational trick. It trades one expensive communication for the ability to run ordinary attention locally.

**Timing:** 2:15–3:35

---

### 5. Now it's just attention (2 states — ~40 seconds)

**State 1:** "Once GPU 0 holds all N tokens for head 0, it runs the standard scaled dot-product attention formula: softmax(Q K^T / √d_k) V. No changes to the math."

**State 2:** "Each GPU runs this in parallel on its head subset. No distributed communication during the attention operation itself. It's local matrix multiply and softmax."

**Key point:** Demystify—attention after the all-to-all is not exotic. It's textbook. The trick was moving data so the textbook formula could run without coordination.

**Timing:** 3:35–4:15

---

### 6. The block rhythm + all-to-all #2 (3 states — ~1:15)

**State 1:** "Inside a Transformer block, the rhythm is: project Q, K, V in sequence-parallel layout [N/P, d]. We're about to do attention, so we call all-to-all to exchange tokens for heads, moving to shape [N, d/P]."

**State 2:** "Compute attention. Now we need to feed-forward next. But the feed-forward layer expects sequence-parallel layout again—one GPU should own one token slice, all hidden dimension. So we call _another_ all-to-all. This one reverses the first: from [N, d/P] back to [N/P, d]."

**State 3:** "Feed-forward and the rest of the block resume in sequence-parallel layout. Then the next block starts again. Two all-to-alls per block—one before attention, one after."

**Memorable mantra:**

> tokens → heads → attention → tokens

**Key point:** This mantra is the entire algorithm. The audience should leave repeating it.

**Timing:** 4:15–5:30

---

### 7. Megatron-SP vs Ulysses: the contribution + scaling (3 states — ~1:25)

**State 1:** "Before Ulysses, the baseline was Megatron-style sequence parallelism. Megatron shards the _attention heads_, not the tokens. GPU 0 computes all tokens for head 0; GPU 1 computes all tokens for head 1. Memory per GPU doesn't shrink with more GPUs—each GPU still holds [N, d/P], all N tokens. Communication to synchronize the results is an all-reduce, which costs 4Nh bytes _independently_ of P."

**State 2:** "Ulysses shards the _sequence_. Each GPU holds [N/P, d] tokens, and communication is per-head all-to-all: 3Nh/P bytes for QKV projection, Nh/P bytes for output gather—total 4Nh/P. This shrinks with P. As you add GPUs, the communication cost per link shrinks too."

**State 3:** "Concrete example: N = 1M tokens, h = 96 heads, d_h ≈ 128. With P = 8: Megatron-SP pays ~4·96·128 = 49K bytes. Ulysses pays ~49K / 8 ≈ 6K bytes per GPU. Scale to P = 64, and the advantage grows. This is why the paper claims >10× communication reduction."

**Key point:** Name the Megatron baseline explicitly. The contrast—sharding heads vs sharding sequence—is the paper's core contribution.

**Timing:** 5:30–6:55

---

### 8. Results, divisibility trap, and ZeRO (3 states — ~1:15)

**State 1:** "The paper reports: ~2× typical throughput improvement (Figure 4), up to 2.5× with careful tuning. 1.2B-parameter GPT model sustains 1M+ token sequences. Communication volume >10× lower than baselines."

**State 2:** "One implementation trap: P must divide h—the number of attention heads. If you have 96 heads and 8 GPUs, 96/8 = 12 heads per GPU: works. If you have 97 heads, it fails. This is not a paper theorem; it's a systems constraint from the all-to-all collective."

**State 3 (Q&A only — do NOT narrate in core timing):** "The paper also sketches ZeRO integration. ZeRO-3 stores 1/P of the model weights on each GPU; Ulysses partitions across both sequence-parallel (P_s) and data-parallel (P_d) dimensions. A 2×2 grid: each GPU stores 1/(P_s · P_d) = 1/4 of parameters, not 1/2 from data-parallelism alone. But weight memory is separate from activation memory; activation memory still shrinks by 1/P_s from sequence parallelism."

**Key point:** Report numbers honestly. Highlight the divisibility constraint as an implementation detail, not a flaw. Defer ZeRO-3 detail to Q&A if it comes up—the core story doesn't need it.

**Timing:** 6:55–8:10 (core); Z eRO state deferred to Q&A.

---

### 9. Takeaway: sequence length is now parallel (2 states — ~1:15)

**State 1:** "Repeat the spine: tokens → heads → attention → tokens. That's the entire system. By temporarily reshaping activations, Ulysses parallelizes the sequence axis, something that wasn't feasible before."

**State 2:** "One final thought: DeepSpeed Ulysses is powerful because it turns sequence length from a memory wall into a distributed layout problem. And layout problems have solutions."

**Closing:** "Thank you. Let's talk about assumptions, limitations, and directions."

**Timing:** 8:10–9:25

---

## Timing summary (core)

| Slide | States | Content | Duration | Cumulative |
|-------|--------|---------|----------|------------|
| 1 | 2 | Context wall | 0:40 | 0:40 |
| 2 | 2 | Missing axis | 0:45 | 1:25 |
| 3 | 2 | Sequence-sharded | 0:50 | 2:15 |
| 4 | 3 | All-to-all #1 | 1:20 | 3:35 |
| 5 | 2 | Local attention | 0:40 | 4:15 |
| 6 | 3 | Block rhythm | 1:15 | 5:30 |
| 7 | 3 | Megatron vs Ulysses | 1:25 | 6:55 |
| 8 | 3 | Results + divisibility | 1:15 | 8:10 |
| 9 | 2 | Takeaway | 1:15 | 9:25 |

**Core timing: 9:25** (leaves ~5 minutes for Q&A and discussion)

---

## Q&A bank

### Why P must divide h (divisibility constraint)

**Question:** "Why can't we have, say, 97 heads with 8 GPUs?"

**Answer:** "During the all-to-all collective, we partition the head dimension evenly across GPUs. Eight GPUs need eight disjoint head groups. If 97 doesn't divide by 8, some GPUs get 12 heads and others get 13, which breaks the all-to-all gather-scatter symmetry in the implementation. It's not a deep algorithmic issue—just a collective's implementation detail. Modern systems are starting to relax this by padding or dynamic scheduling, but the paper assumes exact divisibility."

---

### GQA/MQA and Ulysses (grouped/multi-query attention)

**Question:** "What if we use grouped query attention (GQA) or multi-query attention (MQA)? Does Ulysses still work?"

**Answer:** "GQA has fewer unique KV heads than query heads. This complicates the head-sharding partition. If you have 96 Q heads but only 8 KV heads, and P = 8 GPUs, you'd need to replicate KV heads across multiple GPUs or use a finer-grained shard. The paper doesn't address this; current implementations assume full multihead attention (one KV head per Q head, or nearly so). GQA/MQA would require modified collectives, which the authors leave as future work."

---

### ZeRO-3 in Ulysses: 1/4 vs 1/2 weight memory

**Question:** "The paper mentions 1/4 of the weights per GPU. Isn't ZeRO-3 normally 1/P, which would be 1/8 for P=8?"

**Answer:** "Good catch. ZeRO-3 shards parameters across all devices _uniformly_. Ulysses+ZeRO-3 uses a 2D grid: if you have P_s GPUs for sequence parallelism and P_d for data parallelism, weights are sharded across the P_s × P_d = P_total devices. So each GPU stores 1/P_total of the weights. Example: P_s = 2, P_d = 2 (a 2×2 grid) gives P_total = 4, so each GPU stores 1/4 of weights. But _activation_ memory shrinks only with P_s—the sequence-parallel degree—because that's what determines the token slice size. So activation memory per GPU is 1/P_s, and weight memory is 1/(P_s · P_d). They're independent axes."

---

### Is this FlashAttention?

**Question:** "Is Ulysses the same as FlashAttention?"

**Answer:** "No. FlashAttention is a _local_ attention kernel—it's an efficient algorithm for computing standard attention on a _single_ GPU using IO awareness and tiling. Ulysses is a _distributed_ parallelism strategy across multiple GPUs. They're orthogonal. Ulysses can and should use FlashAttention (or other efficient kernels) on the local GPU attention inside each device after the all-to-all. FlashAttention doesn't solve the multi-GPU communication problem that Ulysses addresses."

---

### What about slow or weak interconnects?

**Question:** "All-to-all sounds expensive. What if my cluster has slow interconnects—would Ulysses still help?"

**Answer:** "Ulysses _requires_ fast all-to-all. If your interconnect is slow, the all-to-all latency will dominate, and you might not see speedups. The paper assumes fast interconnect—NVLink within a node, InfiniBand across nodes. If communication is truly the bottleneck, ring attention (which uses point-to-point communication instead of all-to-all) might be better. Ulysses is a systems contribution that shines on modern high-bandwidth clusters."

---

### Difference between Ulysses and ring attention

**Question:** "How does Ulysses compare to ring attention? Don't they both do sequence parallelism?"

**Answer:** "Ring attention (also called cycle attention in some papers) uses point-to-point p2p communication—token slices rotate around a ring topology. Ulysses uses all-to-all collectives, which require every GPU to exchange with every other GPU simultaneously. Ring attention has lower _latency_ per round trip and can work on weaker interconnects, but may require more communication rounds. Ulysses is faster on high-bandwidth networks because all-to-all is hardware-optimized on modern clusters (NVLink, IB). The paper doesn't directly compare to ring; they focus on the all-to-all advantage on their test clusters."

---

### Why not just use bigger GPUs?

**Question:** "If activation memory is the problem, why not just use bigger GPUs or offload to CPU?"

**Answer:** "Bigger GPUs are expensive and have scaling limits. CPU offloading adds latency—moving gigabytes of activations between GPU and CPU memory is slow. Ulysses lets you use the GPUs you have more efficiently by distributing the computation. It's a pragmatic systems solution for clusters that _are_ already available, rather than betting on new hardware."

---

### Can Ulysses and Megatron SP be combined?

**Question:** "Could you use both Megatron sequence parallelism (head sharding) and Ulysses sequence parallelism (token sharding) at the same time?"

**Answer:** "In principle, you could use a finer-grained partitioning, but it quickly becomes complicated. Ulysses already amortizes the all-to-all cost efficiently. Mixing two different sequence-parallel strategies would multiply the collectives and likely increase synchronization overhead. The paper's design choice—pure token sharding via all-to-all—is simpler and more efficient."

---

### What's the activation memory breakdown?

**Question:** "You said activation memory shrinks by 1/P. What exactly is stored?"

**Answer:** "Activation memory per GPU in Ulysses holds: (1) the Q/K/V projections for the token slice [N/P, d], (2) the intermediate activations in feed-forward [N/P, d × d_ff] (which is quite large), (3) the attention weight matrix during the all-to-all phase [N, d/P] (briefly), and (4) gradient buffers if backpropagating. The dominant term is usually the feed-forward intermediate. Because we reduce N by a factor of P during the sequence-parallel phase, activation memory shrinks roughly by 1/P. GQA and other optimizations further reduce the key-value size."

---
