# Literature snapshot — RoLoRA project positioning (2026-05-20)

This note records the external research context we checked while planning how to push the CS4725 RoLoRA project from a solid reproduction into a top-grade / possible workshop-paper direction. It complements the local canonical sources in `docs/research/`: the RoLoRA paper PDF, the submitted proposal, and the deep-research plan.

## Project-relevant framing

Our submitted proposal commits to reproducing RoLoRA and testing three improvements that preserve RoLoRA's alternating structure:

1. improved initialization for the down-projection matrix `A`,
2. separate learning rates for `A` and `B`,
3. adaptive server-side optimization instead of plain averaging.

The strongest way to make these read as research rather than disconnected tricks is to unify them under one question:

> RoLoRA removes LoRA aggregation error through alternating exact aggregation, but how sensitive is it to phase-specific behavior of the `A` and `B` factors?

Under this framing:

- initialization tests whether the starting `A` subspace controls early convergence and variance;
- separate A/B learning rates test whether the two alternating phases need different optimization scales;
- adaptive server optimization tests whether the active-factor updates should be accumulated differently across rounds.

## Sources checked

### RoLoRA — project ground truth

- Source: OpenReview, *Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA*, NeurIPS 2025 poster.
- Link: <https://openreview.net/forum?id=e8DrPuJekZ>
- Relevance: confirms the accepted RoLoRA artifact, supplementary material, and main claims: alternating LoRA optimization, learning both projection matrices, reduced communication, and RoBERTa-Large / Llama-2-7B experiments.
- Project implication: our reproduction should prioritize the central RoBERTa-Large client-scaling result and Figure-3-style convergence curves before claiming improvements.

### LoRA+ — support for asymmetric A/B learning rates

- Source: Hayou, Ghosh, Yu, *LoRA+: Efficient Low Rank Adaptation of Large Models*.
- Link: <https://arxiv.org/abs/2402.12354>
- Relevance: argues that ordinary LoRA is suboptimal for large-width models when `A` and `B` use the same learning rate, and proposes different A/B learning rates.
- Project implication: our separate-learning-rate direction is well motivated, not arbitrary. The RoLoRA-specific twist is that the factors are optimized in alternating communication rounds rather than simultaneously.

### ADF-LoRA / TAD-LoRA — alternating LoRA under decentralized FL

- Sources:
  - *ADF-LoRA: Alternating Low-Rank Aggregation for Decentralized Federated Fine-Tuning* — <https://arxiv.org/abs/2511.18291>
  - *Stabilizing Decentralized Federated Fine-Tuning via Topology-Aware Alternating LoRA* — <https://arxiv.org/abs/2602.00451>
- Relevance: both papers extend alternating low-rank/factor ideas to decentralized or topology-aware federated settings, highlighting phase-state mismatch, block-wise divergence, topology-induced cross terms, and stability under communication graphs.
- Project implication: broad claims like “we are the first to study alternating LoRA under realistic FL complications” are unsafe. If we discuss partial participation or topology later, we must position it against this literature.

### RD-LoRA — alternating freezing plus routing/decomposition

- Source: OpenReview, *Routing-Deconstructed LoRA in Federated Fine-Tuning*.
- Link: <https://openreview.net/forum?id=6xB2mKOGqx>
- Relevance: builds on alternating freezing to mitigate aggregation noise, then adds routing/decomposition and adaptive aggregation for heterogeneous settings.
- Project implication: adaptive aggregation in our project should be framed narrowly as a lightweight server optimizer for RoLoRA’s active factor, not as a broad new heterogeneous FL framework.

### FedRot-LoRA — rotational misalignment

- Source: *FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA*.
- Link: <https://arxiv.org/abs/2602.23638>
- Relevance: argues that low-rank factorization has rotational invariance, so semantically equivalent client updates can be represented in misaligned factor bases and interfere destructively under factor-wise averaging.
- Project implication: orthogonal/SVD initialization should not be oversold as solving all factor misalignment. It is better framed as testing whether a better shared starting subspace helps RoLoRA’s early A/B phases.

### LA-LoRA — privacy-preserving local alternation

- Source: *Rethinking LoRA for Privacy-Preserving Federated Learning in Large Models*.
- Link: <https://arxiv.org/abs/2602.19926>
- Relevance: studies differential privacy and local alternation, reporting strong gains over RoLoRA in strict DP settings.
- Project implication: do not make DP-RoLoRA the main improvement unless the team deliberately pivots. The DP niche is already crowded and higher-risk.

### SDFLoRA — rank/data heterogeneity and privacy-aware decoupling

- Source: *SDFLoRA: Selective Dual-Module LoRA for Federated Fine-tuning with Heterogeneous Clients*.
- Link: <https://arxiv.org/abs/2601.11219>
- Relevance: targets heterogeneous client ranks/data distributions and privacy-aware separation of global versus local adapter components.
- Project implication: heterogeneous-rank clients are not a good primary novelty direction for this course project; they would distract from the proposal and are already being studied.

### FedMomentum — SVD and momentum-preserving aggregation

- Source: *FedMomentum: Preserving LoRA Training Momentum in Federated Fine-Tuning*.
- Link: <https://arxiv.org/abs/2603.08014>
- Relevance: identifies loss of training momentum in federated LoRA and uses mathematically correct aggregation plus SVD reconstruction/residual handling.
- Project implication: our adaptive server-side optimization should stay lightweight and RoLoRA-specific. Avoid claiming that “server momentum for federated LoRA” is new. Instead ask whether momentum/Adam on the active RoLoRA factor helps or destabilizes alternating phases.

## Strategic conclusion

The broad federated-LoRA improvement space is crowded. For this course project, the strongest and safest 12/10 story is not “we invent a new federated LoRA framework.” It is:

> We reproduce RoLoRA’s core scaling claim, then characterize and improve phase-specific A/B dynamics using small interventions that preserve RoLoRA’s exact alternating aggregation and communication profile.

This gives us a coherent report structure:

1. reproduce the RoLoRA client-scaling result;
2. show diagnostics for A-phase versus B-phase behavior;
3. test initialization, asymmetric learning rates, and active-factor server optimization as targeted interventions;
4. report positive and negative results honestly.

## What this means for experiments

Priority order:

1. Establish a credible RoBERTa-Large reproduction cell before improvement claims.
2. Add diagnostics to log per-round phase, active factor, validation metric, update norm, and optionally factor cosine/subspace movement.
3. Test the smallest improvement grid that can answer the phase-specific question:
   - initialization: default vs orthogonal-A vs SVD/data-informed-A if feasible;
   - learning rates: symmetric vs a small A/B ratio sweep;
   - server optimization: FedAvg vs server momentum/Adam on the active factor only.
4. Prefer one dataset done well over five datasets done weakly.
