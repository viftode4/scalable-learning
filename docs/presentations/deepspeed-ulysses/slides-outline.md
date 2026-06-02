# Group 7 — DeepSpeed Ulysses presentation outline

Goal: make the mechanism understandable from the visuals before the speaker explains it. Presentation slot: Tuesday 9 June 2026, 14:30, Group 7.

The deck is a **9-slide / 18-state visual explainer**. The right arrow advances animation states first, then slides.

---

## Story arc

### 1. Context wall — 2 states

**Visual states:**
- State 1: Short token sequence fits inside one GPU memory boundary.
- State 2: Same sequence grows longer; memory overflow indicator appears.

**Takeaway:** One long document (book, genome, climate data) can be a single training example, making activation memory the bottleneck. Long context requires distributed memory solutions.

**Technical note:** Equations appear after visuals. Frame the bottleneck as peak activation size during forward/backward pass, not parameters.

---

### 2. Missing axis — 2 states

**Visual states:**
- State 1: Labeled parallelism axes: Batch (B), Hidden dimension (H), Layers (L). Token sequence (N) is muted/faded.
- State 2: Token sequence axis highlights/brightens; question mark appears.

**Takeaway:** Existing parallelism covers batch, hidden, and layers. Sequence length N is the unsolved axis. Ulysses targets N as a new parallel dimension.

**Technical note:** Use CSS for this slide (no 3D world needed). Equations show standard parallelism formulas: N × d memory shrinks by 1/P when you parallelize across P GPUs.

---

### 3. Sequence-sharded layout — 2 states

**Visual states:**
- State 1: Four GPUs arranged in a row or grid. Each GPU owns a contiguous token chunk: GPU 0 → tokens [0, N/4), GPU 1 → tokens [N/4, N/2), etc. All head colors are present on each GPU.
- State 2: Zoom or highlight: GPU 0 shows [N/4, d] activation shape. Label: "All heads present, tokens sharded."

**Takeaway:** Before attention, tokens are distributed; each GPU owns 1/P of the token stream. All attention heads are still present on each GPU, so memory per GPU is [N/P, d]. This is memory-efficient, but attention requires global context.

**Technical note:** Use persistent 3D world (three.js) for slides 3–9. Token chunks should have distinct visual identity colors (e.g., red for tokens 0–N/4, green for N/4–N/2, etc.). These colors persist through all slides.

---

### 4. All-to-all layout swap — 3 states

**This is the hero slide. Visuals must be clear and animated.**

**Visual states:**
- State 1: Before all-to-all. GPUs in a ring or grid. GPU 0 shows tokens 0–N/4 in red, both head colors (blue and orange). GPU 1 shows tokens N/4–N/2 in green, both head colors. Label: "GPU 0: [N/4 tokens, d hidden]."
- State 2: Collective exchange in progress. Arrows show data flowing peer-to-peer between all GPU pairs. Visual metaphor: packet rain or mesh of communication lines. Highlight: this is not a broadcast or all-gather; it's a true all-to-all exchange.
- State 3: After all-to-all. GPU 0 now shows all four token colors (red + green + ...) but only one head color (blue). GPU 1 shows all token colors but only the other head color (orange). Label: "GPU 0: [N tokens, d/P hidden], head 0 only."

**Takeaway:** All-to-all is the central algorithm. Tokens are redistributed so each GPU owns the full sequence for a single head subset. The transformation preserves the computation—same data, new layout.

**One-liner:** "Same tokens, different layout."

**Technical note:** This is the most important slide. Spend time on the animation. Make the token colors flow clearly through the exchange. After the visual, overlay KaTeX with the shape transformation: `[N/P, d] --all-to-all--> [N, d/P]`.

---

### 5. Local attention — 2 states

**Visual states:**
- State 1: Zoom into GPU 0 (post all-to-all). Show the full token sequence (all colors) arranged as a sequence, and the head subset highlighted.
- State 2: Attention weight matrix lights up. Shade regions: Query tokens on one axis (full sequence), Key tokens on the other axis (full sequence). Show softmax weights in a heatmap. Label: "Softmax(Q K^T / √d_k) V."

**Takeaway:** After the all-to-all, ordinary local attention runs per GPU. No distributed communication needed during attention itself—each GPU has both Q and K for its head subset.

**Technical note:** Overlay the attention formula after the visual lights up. Emphasize that the math is unchanged; only the data layout changed.

---

### 6. Transformer block recipe — 3 states

**Visual states:**
- State 1: Show the sequence of operations:
  - Project Q, K, V → shape [N/P, d]
  - Labeled: "Sequence-parallel" (token chunks visible on each GPU).
- State 2: All-to-all collective highlighted in the flow. Arrows show data exchange. Output shape [N, d/P] visible on each GPU. Label: "All-to-all: tokens → heads."
- State 3: Attention computation box, then another all-to-all (reverse direction), then feed-forward box. After the reverse all-to-all, shape is [N/P, d] again. Label on reverse all-to-all: "All-to-all: heads → tokens." Final label: "Feed-forward continues sequence-parallel."

**Takeaway:** Inside a Transformer block, the rhythm alternates: Q/K/V in sequence-parallel, all-to-all to heads, attention, all-to-all back to tokens, feed-forward. Two all-to-alls per block.

**Mantra:** Display prominently during or after State 3:
> **tokens → heads → attention → tokens**

**Technical note:** Use animation to walk through the block recipe step-by-step. This is the spine of the algorithm. The audience should leave the presentation repeating this phrase.

---

### 7. Megatron-SP vs Ulysses: the contribution + scaling — 3 states

**Visual states:**
- State 1: Megatron-style baseline. Show GPU 0 with [N, d/P] (all tokens, one head). GPU 1 similarly. Bar chart or annotation: "Megatron-SP: all-reduce after every layer, cost = 4Nh (independent of P)." Label: "Each GPU holds full sequence."
- State 2: Ulysses approach. Show GPU 0 with [N/P, d] (one token slice, all heads). Diagram shows smaller memory footprint. Annotation: "Ulysses: all-to-all per layer, cost = 4Nh/P (shrinks with P)."
- State 3: Concrete scaling example. Show a bar chart:
  - X-axis: P (number of GPUs) = 8, 16, 32, 64.
  - Y-axis: Communication per GPU (bytes).
  - Megatron-SP: flat line at 4Nh.
  - Ulysses: line dropping as 4Nh/P.
  - Annotation: "N = 1M tokens, h = 96 heads. Ulysses: ~49KB (P=8) → ~6KB (P=64). >8× improvement."

**Takeaway:** Megatron shards heads (memory doesn't shrink). Ulysses shards tokens (memory shrinks by 1/P, communication shrinks by 1/P). The paper's core contribution is naming and exploiting token sharding at scale.

**Technical note:** Make the bars or curves visually distinct. Megatron's flat line should contrast sharply with Ulysses's descending curve. This slide teaches the scalability advantage.

---

### 8. Results, divisibility trap, and ZeRO — 3 states

**Visual states:**
- State 1: Benchmark results. Show:
  - "~2× typical throughput improvement (Figure 4); up to 2.5× optimized."
  - "1.2B parameter GPT model, 1M+ token sequences sustained."
  - ">10× communication volume reduction vs SOTA."
  - Optional: training curves or speedup plots from the paper.
- State 2: Divisibility constraint. Label or highlight: "P must divide h (number of attention heads)." Example: "96 heads, 8 GPUs → 12 heads/GPU ✓. 97 heads, 8 GPUs → ✗ (97 ÷ 8 ≠ integer)." Note: "Implementation detail, not a theorem."
- State 3 (Q&A only — do NOT include in core slide narration or timing):
  - ZeRO-3 integration. Show a 2×2 grid of GPUs.
  - Annotation: "Each GPU stores 1/(P_s · P_d) = 1/4 of parameters (not 1/2 from DP alone)."
  - Separate annotation: "Activation memory still shrinks by 1/P_s from sequence parallelism (independent axis)."

**Takeaway:** The paper delivers strong reported results. The divisibility constraint is a systems implementation detail. ZeRO-3 detail is a bonus—mention only if Q&A arises.

**Technical note:** For state 3, dim the visual slightly or mark it as "Advanced / Q&A." Do not narrate it during the core 9:25-minute talk.

---

### 9. Takeaway: sequence length is now parallel — 2 states

**Visual states:**
- State 1: Repeat the mechanism diagram. Show the full four-step cycle:
  - tokens (GPU shards) → all-to-all → heads (GPU shards) → attention (local) → all-to-all → tokens.
  - Render the cycle as a loop or circle to emphasize repetition.
  - Label prominently: "tokens → heads → attention → tokens."
- State 2: Final message. Display the mantra again. Add three Q&A starter prompts (optional, for silence-breaking):
  - "How does this compare to ring attention?"
  - "What if my cluster has weak interconnect?"
  - "Does P need to divide h exactly?"

**Takeaway:** Ulysses makes sequence length a parallel dimension by temporarily exchanging the activation layout. The mechanism is elegant because it turns a memory problem into a layout problem, and layout problems have efficient solutions (all-to-all collectives).

**Closing line (verbal):** "DeepSpeed Ulysses is powerful because it transforms sequence length from a memory wall into a distributed systems problem." Then: "Thank you. Questions?"

**Technical note:** Keep state 2 clean and readable. The three prompts are visual insurance against Q&A silence—if the audience is quiet, you can read them aloud to seed discussion.

---

## Technical notes on 3D world (three.js)

**Scope:** Slides 3–7 and 9 use a persistent 3D GPU/token visualization. Slide 2 uses CSS only.

**Visual elements:**
- GPUs: boxes or cubes with labels (GPU 0, GPU 1, ..., GPU P−1).
- Tokens: colored blocks or rods, with distinct hues per token range (red, green, blue, yellow, etc.).
- Heads: visual attributes (e.g., layered shading or patterns) to represent head dimensions.
- All-to-all: animated arrows or packet streams flowing between GPUs.

**Persistence:** The token colors established in slide 3 should remain consistent through all slides. A viewer should immediately recognize "red tokens" from slide 3 when they reappear in slide 4's post-all-to-all state.

**Transitions:** Smooth camera pans and zooms between states. Avoid jarring cuts; let the 3D space guide the eye.

---

## Equations layer (KaTeX, overlaid after visuals)

Equations appear _after_ the corresponding visual animates, not before. This keeps focus on the mechanism.

**Slide 3:** `activation_memory = [N/P, d]` (per GPU)

**Slide 4:** `[N/P, d] --all-to-all--> [N, d/P]` (shape transformation)

**Slide 5:** `Attention = softmax(Q K^T / √d_k) V` (unchanged formula)

**Slide 6:** (Implicit in the flow diagram, no separate equation needed; or label the all-to-all shapes inline.)

**Slide 7:**
- Megatron-SP: `comm = 4Nh` (per GPU per layer)
- Ulysses: `comm = 4Nh/P` (per GPU per layer)

**Slide 8:** Constraint: P divides h. (Not an equation, a design constraint.)

**Slide 9:** (Implicit mantra; no additional equations.)

---

## Rehearsal spine

Repeat this phrase several times during the talk (especially around slides 4, 6, and 9):

> **tokens → heads → attention → tokens**

This is the mechanism the audience should remember. Every state transition in slides 3–7 is a step in this cycle. By slide 9, saying it should feel like describing a familiar dance.

---

## Timing and pacing

- **Slides 1–3:** 2:15 (establish motivation and layout).
- **Slides 4–6:** 3:10 (the algorithm itself; anchor the mantra here).
- **Slides 7–8:** 2:40 (contribution and results; divisibility trap; defer ZeRO to Q&A).
- **Slide 9:** 1:20 (closing; Q&A prompts).

**Total core:** 9:25. This leaves ~5 minutes for questions, with time to dive into divisibility, GQA, ZeRO-3, ring attention, and interconnect trade-offs.

---
