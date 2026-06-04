# DeepSpeed Ulysses deck — simplified conceptual spec

Goal: explain the method conceptually before showing math. Each slide carries one idea and uses low-density drawings. The visual invariant is a toy tensor with `N=8` tokens, `H=4` head groups, and `P=4` GPUs.

1. Title: token-sharded → head-sharded → attention → token-sharded.
2. Long sequence problem: activations are tall token-row matrices.
3. Attention is global: one head needs an `N × N` score matrix.
4. Conflict: token sharding fragments each head across GPUs.
5. Naive all-gather: replicating everything is wasteful.
6. Ulysses core idea: swap the owner rule with all-to-all.
7. Diagram legend: rows=tokens, columns=heads, border=GPU owner.
8. Start layout: local shape `[N/P, d]`.
9. First all-to-all: token owners send head slices to head owners.
10. After first all-to-all: local shape `[N, d/P]`.
11. Attention works: GPU2 can compute full h2 scores locally.
12. Values/context: softmax weights mix V into context.
13. All heads parallel: each GPU runs its head group.
14. Context problem: head-sharded context is wrong for MLP.
15. Second all-to-all: send context slices back to token owners.
16. Restored layout: local shape `[N/P, d]` again.
17. Whole layer rhythm: local → exchange → local → exchange → local.
18. Communication intuition: all-to-all sends useful slices, not replicated tensors.
19. Minimal formula: `(3Nh + Nh) / P = 4Nh/P`.
20. Scaling story: grow `P` with `N` so local `N/P` stays manageable.
21. Scope and evidence: what Ulysses solves, assumes, and does not remove.
22. Takeaway: attention needs all tokens for one head, not all heads everywhere.
