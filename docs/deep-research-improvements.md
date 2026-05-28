# Practical Improvements to RoLoRA: A Survey of Federated and Centralized LoRA Tricks

## TL;DR

- **The three highest-leverage, easy-to-bolt-on changes for RoLoRA are: (a) replace the random Gaussian init of A with PiSSA/SVD-based or orthogonal initialization (RoLoRA's own theory shows FFA-LoRA's error scales with ‖a⁰ − a*‖, so a smarter A-init should sharpen the FFA-LoRA → RoLoRA gap), (b) adopt LoRA+ asymmetric learning rates within each half-round (λ = η_B/η_A ≈ 16 for RoBERTa-Large; ≈ 2–4 for Llama-7B per Hayou, Ghosh, Yu, ICML 2024), and (c) replace LoRA's α/r scaling with rsLoRA's α/√r so higher ranks actually help in the alternating regime.** These three are pure drop-ins that touch only initialization, the per-step optimizer, and the α scaling — none change RoLoRA's alternating schedule or its halved communication cost.
- **At small-model / RoBERTa-Large scale (≈355M params, the regime RoLoRA emphasizes), the empirically validated pickups are concrete: LoRA+ "improves performance (1%–2% improvements) and finetuning speed (up to ~2× SpeedUp), at the same computational cost as LoRA"; PiSSA outperforms LoRA on 14 of 16 GLUE tasks for RoBERTa-large/DeBERTa-v3-base; FFA-LoRA's own Appendix A.8 ablation finds orthogonal-A initialization beats Kaiming on QNLI by ~0.32 pp (91.84 → 92.16); FedSVD reports +1.29 pp avg over FFA-LoRA without DP and +8.77/+9.63 pp under DP at ε ∈ {6, 3} on GLUE.** None of these are mutually exclusive with RoLoRA's alternation.
- **Avoid heavyweight reframings (DoRA, GaLore, AdaLoRA's SVD reparameterization, FlexLoRA's server-side full-matrix SVD) at first pass: they either inflate communication, break RoLoRA's "only one of {A,B} is trainable per round" invariant, or — in the case of DoRA's magnitude vector — introduce a third non-low-rank parameter that breaks the alternating structure.** Heterogeneous-rank support (LoRA-A², HetLoRA, FlexLoRA, FLoRA, FSLoRA, raFLoRA) is achievable but strictly second-priority because RoLoRA's design assumes shared rank across clients.

---

## Key Findings

### 1. Federated-LoRA improvements (2023–2025) most directly relevant to RoLoRA

| Method | Year/Venue | Core idea | What to import into RoLoRA |
|---|---|---|---|
| **FFA-LoRA** (Sun et al., arXiv 2403.12313) | ICLR 2024 | Freeze A, train B only | RoLoRA's direct baseline; its Appendix A.8 init ablation (orthogonal vs Kaiming vs SVD-of-W₀) is directly reusable. |
| **FedSA-LoRA** (Guo et al., arXiv 2410.01463) | ICLR 2025 | Train both A and B locally, but **share only A** with the server, because A learns "general" and B learns "client-specific" knowledge | Strong evidence that A-aggregation carries the global signal; informs which RoLoRA round to weight more under non-IID. |
| **FedEx-LoRA** (Singhal, Ponkshe, Vepakomma, arXiv 2410.09432) | ACL 2025 (Oral) | Add explicit residual ΔW_res = avg(BᵢAᵢ) − (avg B)(avg A) onto the frozen base weights at the server | Drop-in fix for *jointly* updated rounds — RoLoRA already eliminates this product-of-averages bias by alternating, so FedEx-LoRA is mostly redundant with RoLoRA. |
| **FedSVD** (Lee et al., arXiv 2505.12805) | NeurIPS 2025 | Server computes BA, refactors via SVD; A becomes orthonormal right singular vectors | **Could replace RoLoRA's A-update half-round with a server-side SVD refresh** — provides the orthonormal A that FFA-LoRA's own ablation hints at. |
| **FRLoRA** (Yan et al., OpenReview iclr.cc/.../28955) | ICLR 2025 | Sum BA into the frozen base weights at each round, reinitialize LoRA | Mitigates client drift via residual update; combinable with RoLoRA's alternation. |
| **LoRA-A²** (Koo et al., arXiv 2410.22815) | ACL 2025 | Same alternating-freeze idea as RoLoRA + adaptive per-rank importance selection | Most natural "next step" for RoLoRA: add LoRA-A²'s adaptive rank selection inside the alternating schedule. |
| **FLoRA / FlexLoRA / HETLORA / FSLoRA / raFLoRA** | NeurIPS 2024 / ICLR 2024 / EMNLP 2024 / ICML 2025 / 2026 preprint (arXiv 2602.13486) | Heterogeneous ranks across clients via stacking, SVD truncation, sketching, or rank-partitioned aggregation | Required if RoLoRA must support clients with different ranks; not free — FlexLoRA's SVD on full ΔW is expensive, FLoRA's stacking inflates downlink linearly in N, FSLoRA/raFLoRA are the most communication-efficient. |
| **LoRA-FAIR** (Bian et al., arXiv 2411.14961) | ICCV 2025 | Server-side correction term + initialization refinement to handle aggregation bias and "client-side initialization lag" | Informs how to redistribute A back to clients between RoLoRA's alternating rounds. |
| **SLoRA** (Babakniya, Elkordy, Ezzeldin, Liu, Song, El-Khamy, Avestimehr, arXiv 2308.06522) | Oral, FL@FM-NeurIPS'23 Workshop | Two-stage: sparse fine-tune warm-up, then SVD-decompose to initialize LoRA | Useful as a one-time warm-start before launching RoLoRA's alternating rounds; "SLoRA achieves performance comparable to full fine-tuning, with significant sparse updates with ~1% density while reducing training time by up to 90%." |
| **DEeR** (arXiv 2410.12926) | MICCAI 2024 | Alternating min + DP noise regulator | Independent corroboration that alternating helps under DP; suggests pairing RoLoRA with explicit DP-noise control. |
| **FedAdamW / FedAdam / FedYogi** (arXiv 2510.27486 / Reddi et al., ICLR 2021) | 2025 / ICLR 2021 | Server-side adaptive optimizer with first/second-moment state | Drop-in replacement for the simple averaging on either of RoLoRA's two halves. |
| **FedMomentum** (arXiv 2603.08014) | 2026 preprint | Preserve LoRA "training momentum" across rounds via momentum-aware SVD | Addresses the precise failure mode RoLoRA's alternation partially fixes — combinable. |

### 2. Centralized LoRA tricks that translate cleanly to RoLoRA's alternating regime

| Trick | Touches | Verbatim claim | Translates to RoLoRA? |
|---|---|---|---|
| **LoRA+** (Hayou, Ghosh, Yu, ICML 2024, arXiv 2402.12354) | Per-step optimizer | "improves performance (1%–2% improvements) and finetuning speed (up to ~2× SpeedUp), at the same computational cost as LoRA" with η_B = λ·η_A, λ ≈ 2⁴ = 16 for RoBERTa | Yes, trivially: in B-rounds use η_B, in A-rounds use η_A. No state coupling because the matrices are never trained simultaneously. |
| **rsLoRA** (Kalajdzievski 2023, arXiv 2312.03732) | Scaling factor α/r → α/√r | Stabilizes gradients at higher ranks; lets larger ranks actually help | Yes — orthogonal to alternation; one-character change. |
| **PiSSA** (Meng, Wang, Zhang, NeurIPS 2024 Spotlight, arXiv 2404.02948) | Initialization | A,B initialized with principal SVD components; "outperforms LoRA on all 5 common benchmarks"; on GSM8K, Mistral-7B fine-tuned with PiSSA reaches 72.86%, "outperforming LoRA's 67.7% by 5.16%"; on RoBERTa-large/DeBERTa-v3-base it outperforms LoRA on 14 of 16 GLUE tasks | Yes — replaces both A⁰ and B⁰. RoLoRA's alternating schedule is initialization-agnostic. |
| **LoftQ** (Li et al., ICLR 2024, arXiv 2310.08659) | Quant-aware init | Iterative SVD to find A,B that minimize ‖W − Q − BA‖_F | Yes; especially for federated edge clients running 4-bit base weights. |
| **OLoRA** (Büyükakyüz 2024, arXiv 2406.01775) | Initialization | QR decomposition of W₀ to derive orthonormal A,B; faster convergence | Yes; cheaper than SVD; aligns with FFA-LoRA's own Appendix A.8 finding that orthogonal-init A wins on QNLI (91.84 → 92.16 mean). |
| **LoRA-XS** (Bałazy et al., NeurIPS 2024, arXiv 2405.17604) | Architecture | Freeze SVD-derived A,B, learn r×r matrix R between them; >100× param reduction at 7B; "with a rank of 16, LoRA-XS achieves better accuracy than VeRA while having 2.5x less trainable parameters" on RoBERTa-large GLUE | Partial: incompatible with RoLoRA's premise that A and B are alternately trained — but the idea of fixing A,B at SVD components and learning a small bridge matrix could become an RoLoRA "third half-round" for ultra-low-bandwidth clients. |
| **LoRA-GA** (Wang, Liang, Wang, NeurIPS 2024, arXiv 2407.05000) | Init | Aligns first-step low-rank gradient with full-FT gradient; "on the subset of the GLUE dataset with T5-Base, LoRA-GA outperforms LoRA by 5.69% on average. On larger models such as Llama 2-7B, LoRA-GA shows performance improvements of 0.34, 11.52%, and 5.05% on MT-bench, GSM8K, and Human-eval, respectively." | Yes for LoRA-GA initialization. Caveat: needs a calibration gradient that may leak signal in federated; use server-side public proxy data only. |
| **LoRA-Pro** (Wang, Liang, He, Wang, Tan 2024, arXiv 2407.18242) | Per-step grad | Closed-form gradient rescaling so that BA's "equivalent gradient" tracks the full-FT gradient | Yes; simplifies dramatically inside RoLoRA half-rounds (only one of ∇A, ∇B is nonzero, so the correction reduces to a scalar reweighting). |
| **AdaLoRA** (Zhang et al., ICLR 2023, arXiv 2303.10512) | Rank schedule | SVD-form parameterization with importance-based singular-value pruning | Partial: AdaLoRA's PΛQ form means three things to update; clashes with RoLoRA's binary alternation. Could be reused inside one of the two half-rounds at the cost of complexity. |
| **DoRA** (Liu, Wang, Yin, Molchanov et al., ICML 2024 Oral, arXiv 2402.09353) | Architecture | Magnitude–direction decomposition; "+3.7/+1.0 on Llama 7B/13B, +2.9 on Llama 2 7B, and +4.4 on Llama 3 8B" common-sense reasoning | Adds a magnitude vector m that is *not* low-rank. Either keep m always-trainable (cancels RoLoRA's halved comm) or add a third "M-round". Useful but heavier. |
| **VeRA** (Kopiczko, Blankevoort, Asano, ICLR 2024, arXiv 2310.11454) | Architecture | Tied frozen random A,B + learned per-layer scaling vectors; ~10× fewer params than LoRA | Partial: drastically cuts comm but doesn't benefit from alternation (B is frozen). FedSA-VeRA already exists. |
| **GaLore** (Zhao et al., ICML 2024, arXiv 2403.03507) | Optimizer | Project full gradients into a low-rank subspace updated periodically via SVD; full-parameter learning at LoRA-like memory | Misaligned: GaLore trains the *full* model in low-rank-projected gradient space; in federated PEFT the goal is the opposite — keep the trainable surface low-rank. |
| **LoRA-the-Explorer (LTE)** (Huh et al., ICLR 2024, arXiv 2402.16828) | Distributed pre-training | Parallel LoRA heads merged periodically | Mirrors federated LoRA but for pre-training; key insight: "even when trained in parallel, LoRA heads maintain orthogonality throughout the training process." |

### 3. What the RoLoRA paper actually reports (empirical anchor)

From the NeurIPS 2025 paper (arXiv 2502.01755), on RoBERTa-Large with 50 clients and rank=4 (the "stress" regime in the abstract figure), QQP accuracies are: LoRA 77.60±1.47, FFA-LoRA 78.44±0.41, **RoLoRA 85.71±0.18** — a +7.27 pp gain over FFA-LoRA. GLUE averages across SST-2/QNLI/MNLI/QQP/RTE: LoRA 70.72, FFA-LoRA 76.48, **RoLoRA 85.81** at rank=4; and 64.03 / 72.46 / **86.27** at rank=8. Theorem 5.4 proves exponential convergence to the global optimum for rank-1 federated linear regression; Proposition 5.5 proves FFA-LoRA's residual error is lower-bounded by ‖a⁰ − a*‖·‖b*‖. **This lower-bound is the lever for many of the recommendations below: any improvement that reduces ‖a⁰ − a*‖ (i.e., a smarter A initialization) tightens the FFA-LoRA bound and accelerates RoLoRA's first A-round.**

---

## Details — Catalog of Improvements with Adaptation Notes

### A. Initialization changes (highest expected value, lowest implementation cost)

**A1. PiSSA-style SVD initialization for A and B (Meng, Wang, Zhang, NeurIPS 2024).** PiSSA initializes A with the top-r right singular vectors of W₀ and B with U·Σ_r, putting the high-energy components inside the trainable adapter and the residual into a frozen W^res. *Adaptation to RoLoRA:* apply PiSSA once at round 0, then run the standard alternating schedule. Because the residual is folded into the (still frozen) base weights, RoLoRA's halved communication is preserved. *Expected gain:* 0.5–2 pp on GLUE (small models) and a measurable speedup in early rounds — exactly the regime where RoLoRA's robustness gap with FFA-LoRA is widest.

**A2. Orthogonal/QR (OLoRA) initialization of A.** FFA-LoRA's own Appendix A.8 ablation reports for QNLI: Kaiming init 91.84% (var 0.38), **orthogonal init 92.16% (var 0.83)**, SVD-of-W₀ init 91.50% (var 0.59). Sun et al. explicitly write that "matrix with orthogonal initialization seems to perform slightly better than the existing approach. However, the performance gap is not significant enough for a definitive answer." It is nevertheless a free win, and FedSVD's analysis makes the same point — orthonormal rows of A bound the gradient norms of B. *Adaptation:* replace `init.kaiming_uniform_(A)` with `nn.init.orthogonal_(A)`.

**A3. Server-side SVD refresh of A (FedSVD-flavored).** FedSVD (Lee et al., arXiv 2505.12805, NeurIPS 2025): "FedSVD achieves the highest average accuracy, outperforming the second-best baseline (FFA-LoRA) by +1.29 percentage points" on GLUE without DP, and "the average gain of FedSVD over FFA-LoRA increases substantially in the DP settings, i.e., from +1.29 pp without privacy constraints to +8.77 pp with ε = 6. Even under a stricter privacy budget (ε = 3) … our method still achieves an accuracy improvement of +9.63 pp." *Adaptation to RoLoRA:* substitute the A-training half-round with a server-side SVD refresh of A from aggregated B. Caveat: you no longer *learn* A from local gradients; choose this variant only when DP noise dominates.

**A4. LoRA-GA gradient-aligned initialization (Wang et al., NeurIPS 2024).** Reported gains on T5-Base GLUE subset are +5.69% average over LoRA, and on Llama 2-7B +0.34 / +11.52% / +5.05% on MT-bench / GSM8K / Human-eval. *Adaptation:* one-time pre-round step. Caveat in federated: the calibration gradient must come from some client(s); only use if a public proxy dataset is available, or compute on the server with public data.

**A5. SLoRA two-stage warm-start (Babakniya et al., FL@FM-NeurIPS'23 Oral).** Stage 1: sparse fine-tuning at ~1% density. Stage 2: SVD-decompose the aggregated update into A,B as the LoRA initialization. Reported: "SLoRA achieves performance comparable to full fine-tuning, with significant sparse updates with ~1% density while reducing training time by up to 90%." *Adaptation:* combine with RoLoRA's stage-2 alternation. Worthwhile under severe non-IID.

### B. Per-step optimizer changes (cheap, drop-in)

**B1. LoRA+ asymmetric learning rates (Hayou, Ghosh, Yu, ICML 2024).** From the paper: "we set the learning rate of B to be λ× that of A, where λ ≫ 1 is fixed." Empirically optimal λ values: "with Init[2], we found that generally setting a ratio of λ = ηB/ηA ≈ 2⁴ improves performance for Roberta," while "with Init[1], we found that the optimal ratio is smaller and is of order 2² – 2³"; "for Llama experiments, it seems that a ratio of order 2¹ – 2² is optimal." Reported GLUE numbers (RoBERTa-base, FP16, α = r = 8): MNLI 86.5 vs 85.5, QQP 89.1 vs 88.5, SST-2 94.7 vs 94.0, QNLI 92.1 vs 90.9. *Adaptation to RoLoRA:* use η_B = 16·η_A in B-rounds (RoBERTa-Large) and η_A in A-rounds. No state synchronization issues because the two matrices are never trained simultaneously — **the cleanest possible match between LoRA+ and RoLoRA's structure.**

**B2. rsLoRA scaling α/√r instead of α/r (Kalajdzievski 2023, arXiv 2312.03732).** "The gradients do not collapse, and training with higher ranks increases performance." Essential if RoLoRA wants to claim accuracy gains from raising rank — RoLoRA's empirical bump from rank 4 → 8 (avg 85.81 → 86.27) is small partly because of the un-rescaled α.

**B3. LoRA-Pro gradient rescaling (Wang et al. 2024, arXiv 2407.18242).** Closed-form per-step rescaling of ∇A and ∇B such that the "equivalent gradient" (∇B·A + B·∇A) tracks the full-FT gradient. *Adaptation to RoLoRA:* simplifies dramatically because in any half-round only one of {∇A, ∇B} is nonzero. AltLoRA (arXiv 2505.12455) already shows that "alternating projections" achieves a similar approximation — a near-relative of RoLoRA. Try LoRA-Pro inside RoLoRA's two halves as a follow-up experiment.

**B4. Server-side adaptive aggregation (FedAdam / FedAdamW / FedYogi).** Reddi et al., ICLR 2021; FedAdamW (arXiv 2510.27486) "aligns local updates with the global update using both a local correction mechanism and decoupled weight decay to mitigate local overfitting." *Adaptation:* server keeps two optimizer states — one for A-rounds, one for B-rounds. Cheap; largest expected gains under non-IID and partial participation.

**B5. Momentum / variance reduction (SCAFFOLD-M, FedGLOMO).** Huang et al., "Momentum Benefits Non-IID Federated Learning Simply and Provably" (ICLR 2024, arXiv 2306.16504), state: "We show that incorporating momentum allows FedAvg and its variance-reduced extension to converge under unbounded data heterogeneity, even using constant local learning rates." *Adaptation to RoLoRA:* maintain server-side momentum on whichever matrix is being aggregated this round. Direct fix for the high-variance failure mode in RoLoRA's 50-client/rank-4 column.

### C. Architectural / aggregation changes (medium cost)

**C1. FedSA-LoRA's selective A-aggregation (Guo et al., ICLR 2025).** Their Lemma 1 proves that for a linear regression objective, the optimal A* "is independent of the input data distribution, while B* is related to the input data distribution captured by E[xxᵀ]. This indicates that A is responsible for learning general knowledge, while B focuses on modeling client-specific knowledge." Empirically (RoBERTa-large, RTE, 3 clients with Dirichlet(0.5), measured by mean pairwise cosine similarity across clients): A's similarity stays at ~0.99–1.00 across IID and non-IID, while B's drops markedly under non-IID. GLUE Table 1 (RoBERTa-large, rank=8, α=16, 3 clients, 1000 rounds): FedSA-LoRA averages 90.43 vs FFA-LoRA 89.39 — **+1.04 pp gain**. *Adaptation to RoLoRA:* keep B fully personal (do not broadcast back) under high heterogeneity. Converts RoLoRA into a personalization-aware variant.

**C2. FedEx-LoRA residual correction.** RoLoRA's alternation already eliminates the product-of-averages bias — within an A-round B is identical across clients, so avg(BᵢAᵢ) = (avgB)(avgA). FedEx-LoRA is therefore **redundant with RoLoRA**.

**C3. FRLoRA residual fusion into base weights (Yan et al., ICLR 2025).** *Adaptation to RoLoRA:* combinable as a periodic "fold-in" every K rounds. Caveat: server-side base weights drift away from the original pretrained checkpoint, breaking quantization compatibility.

**C4. Heterogeneous-rank support.** Trade-offs:
- **FLoRA** (NeurIPS 2024): stacking inflates downlink linearly in N.
- **FlexLoRA** (Bai et al.): server-side full-matrix SVD; expensive on large LLMs.
- **HetLoRA** (Cho et al., EMNLP 2024): zero-pad to max rank, average, truncate; "FSLoRA outperforms HetLoRA in both computation and memory cost" (FSLoRA paper).
- **FSLoRA** (ICML 2025 preprint, arXiv 2501.19389): single global LoRA + binary sketch indices per client; cheapest comm.
- **raFLoRA** (2026 preprint, arXiv 2602.13486): rank-partitioned aggregation that prevents "rank collapse."
- **LoRA-A²** (Koo et al., ACL 2025): closest cousin to RoLoRA — "alternates between freezing LoRA modules B and A in each round" + adaptive rank selection. **Single most direct, code-compatible upgrade to RoLoRA** if heterogeneous ranks are needed.

**C5. LoRA-XS bridge matrix.** Incompatible with RoLoRA's binary-state premise, but can be added as a third half-round dedicated to communicating only R (r×r) during low-bandwidth windows.

**C6. DoRA magnitude–direction decomposition (Liu et al., ICML 2024 Oral).** *Adaptation to RoLoRA:* introduce an "M-round" where only m is trained and aggregated. Adds ~d_out trainable scalars per layer (a vector, not a matrix). Caveat: m is not low-rank, so it is communicated in full each M-round — verify the comm budget still beats vanilla LoRA.

### D. Non-IID, drift, and partial-participation tricks specifically helpful for RoLoRA

**D1. Proximal regularization (FedProx-style) on the *active* matrix only.** Add μ/2 ‖A − A_global‖² in A-rounds, μ/2 ‖B − B_global‖² in B-rounds. Cheap; addresses the high-variance behavior in RoLoRA's RTE 50-client column.

**D2. SCAFFOLD-style control variates for one matrix at a time.** State doubles vs SCAFFOLD but is half the size of full-LoRA SCAFFOLD because only one of {A,B} is active.

**D3. Periodic refresh / two-stage SLoRA warm-start** (see §A5).

**D4. Reservoir / partial-participation sampling.** Standard FL practice; pair with momentum (D5).

**D5. Client momentum (FedSTEPH2 / FedGLOMO style).** Lets RoLoRA tolerate higher data heterogeneity without diminishing local learning rates.

### E. Tricks specifically validated at small (~300M) RoBERTa-Large scale

- **LoRA+** (η_B = 16 η_A): MNLI 86.5 vs 85.5, QQP 89.1 vs 88.5, SST-2 94.7 vs 94.0 (RoBERTa-base, GLUE).
- **PiSSA**: outperforms LoRA on 14/16 GLUE tasks (RoBERTa-large, DeBERTa-v3-base).
- **LoRA-XS**: "with a rank of 16, LoRA-XS achieves better accuracy than VeRA while having 2.5x less trainable parameters" (RoBERTa-large GLUE).
- **OLoRA / orthogonal init**: FFA-LoRA Appendix A.8 reports +0.32 pp on QNLI vs Kaiming.
- **rsLoRA**: enables rank scaling beyond ~8 to actually pay off — relevant since RoLoRA tests rank ∈ {4, 8} but reports limited gain at 8.
- **LoRA-GA**: "+5.69% average over LoRA" on T5-Base GLUE subset.
- **AdaLoRA**: gains over LoRA at low budgets on RoBERTa-base GLUE — but its three-matrix PΛQ form clashes with RoLoRA's binary alternation.

### F. Direct answers to the three RoLoRA-specific questions in the brief

- **Does orthogonal init of A help?** FFA-LoRA's own Table 9 says yes by ~0.32 pp on QNLI; FedSVD's analysis shows orthonormal rows of A bound ‖∇B‖ under DP. **Yes — orthogonal-initialize A.**
- **Does SVD-based init (PiSSA / LoftQ) work in alternating optimization?** No published evidence in the federated alternating setting, but RoLoRA's Proposition 5.5 lower-bounds FFA-LoRA's residual error by ‖a⁰ − a*‖, implying any A-init closer to the eventual minimizer (PiSSA's principal-direction A) reduces both FFA-LoRA's *and* RoLoRA's first-A-round gap. **Try PiSSA once at round 0, then run RoLoRA unchanged.**
- **Different LR for A vs B?** LoRA+ shows +1–2 pp on RoBERTa GLUE; in RoLoRA's alternating regime there is no per-step coupling, so the trick is even cleaner to apply. **Use η_B = 16 η_A on RoBERTa-Large.**

---

## Recommendations (staged, with thresholds)

**Stage 1 — Pure drop-ins (ship today, 2–4 lines of code each):**
1. **PiSSA initialization at round 0.** Replace Kaiming(A), zeros(B) with `peft.init_lora_weights="pissa"` (already merged into `peft` main). Re-run RoLoRA's GLUE/RoBERTa-Large + Llama-2-7B sweeps. **Threshold to escalate:** if PiSSA does not provide ≥0.3 pp on the 50-client/rank-4 average over RoLoRA's reported 85.81, fall back to OLoRA (orthogonal QR init), which is initialization-cheaper.
2. **rsLoRA scaling α/√r.** Single-line change. **Threshold:** if scaling rank from 4 → 16 still yields negligible gain after this fix, do not invest further in rank-as-knob.
3. **LoRA+ asymmetric learning rates.** η_B = 16·η_A on RoBERTa-Large, η_B = 4·η_A on Llama-2-7B. **Threshold:** ≥1 pp average on GLUE 50-client/rank-4 — published LoRA+ deltas predict this.
4. **Orthogonal initialization of A.** Free win; subsumed by 1 if you adopt PiSSA.

**Stage 2 — Aggregation & optimizer upgrades (1–3 days of work):**
5. **Server-side FedAdamW** in place of FedAvg, with separate moment buffers for A-rounds and B-rounds. **Threshold:** convergence-curve speedup ≥1.5×.
6. **FedProx-style proximal term** on the active matrix only, μ ∈ {0.001, 0.01, 0.1}. Target the high-variance regime (50 clients, rank 4). **Threshold:** reduce per-task std-dev (e.g., RoLoRA's RTE ±2.88) by ≥30%.
7. **Server momentum** on whichever matrix is being aggregated this round.

**Stage 3 — Architectural follow-ups (1–2 weeks of work; pick at most one initially):**
8. **FedSA-LoRA-style "B is personal":** train B locally without aggregation in B-rounds; aggregate only A in A-rounds. Best for high-heterogeneity / many-client deployments. **Threshold:** must beat the personalization baseline (FedDPA-LoRA-style) by ≥1 pp.
9. **LoRA-A² adaptive per-rank importance mask** inside RoLoRA's halves. Best path to genuine heterogeneous-rank support without server-side SVD.
10. **FedSVD's server-side SVD refactor of A** as a substitute for the A-update half-round, *only* under DP noise (matches FedSVD's published +9.63 pp at ε = 3 over FFA-LoRA).
11. **DoRA "M-round"** if magnitude–direction decomposition's reported +3–4 pp on Llama justifies adding a third per-round transmission.

**Stage 4 — Defer / avoid:**
- **GaLore** (training paradigm mismatch with PEFT).
- **AdaLoRA's PΛQ reparameterization** (clashes with binary alternation; LoRA-A² is the cleaner adaptive-rank route).
- **FlexLoRA's full-matrix server-side SVD** (compute cost prohibitive at 7B+).
- **FLoRA's stacking aggregator** (downlink scales linearly in N; do not pair with 50-client experiments).

---

## Caveats

- **Most "improvements" listed have not been benchmarked specifically inside RoLoRA's alternating optimization.** Their effectiveness is published in centralized or non-alternating federated settings. Mechanical translations are straightforward but may have second-order interactions (e.g., LoRA+ assumes Adam; rsLoRA changes effective gradient magnitudes that LoRA+ ratio tunings depend on — re-tune jointly).
- **RoLoRA's headline "+8 pp at 50 clients/rank 4" gap is partly an artifact of LoRA collapsing in that regime** (LoRA's MNLI std ±15.07 indicates near-failure). Any improvement on top of RoLoRA must be evaluated in less degenerate regimes (e.g., 3–20 clients) where RoLoRA's own gap to LoRA is small (88.28 vs 88.20 at 3 clients/rank 4); the headroom for additional tricks is correspondingly thinner.
- **Initialization tricks (PiSSA, LoftQ, OLoRA, LoRA-GA) require server access to the pretrained W₀ and possibly calibration data.** This may be incompatible with strict cross-silo deployments. SVD on a 7B model takes seconds with fast SVD libraries (PiSSA's reported runtime); on memory-constrained servers, OLoRA's QR is preferable.
- **Heterogeneous-rank support (LoRA-A², HetLoRA, FlexLoRA, FLoRA, FSLoRA, raFLoRA) is a separate axis from alternation.** RoLoRA assumes shared rank. Combining is non-trivial because the alternation invariant (server has the same A or B copy across all clients) breaks under per-client ranks unless using a sketching scheme (FSLoRA) that preserves a single global module.
- **Under differential privacy, FFA-LoRA's argument that freezing one matrix avoids quadratic noise amplification is real.** RoLoRA's alternation does *not* automatically inherit that benefit when both halves use DP-SGD — verify empirically before assuming RoLoRA + DP-SGD outperforms FFA-LoRA + DP-SGD. FedSVD's design specifically targets this regime.
- **DoRA's magnitude vector and AdaLoRA's PΛQ form both break the "binary state" property** (only A or only B trainable per round) that gives RoLoRA its halved communication. Adapting them requires either trainable-extra-state in *every* round (cancelling RoLoRA's comm advantage) or a third half-round (slowing convergence).
- **The empirical evidence for orthogonal A-init is weak at scale.** FFA-LoRA's own ablation calls the gap "not significant enough for a definitive answer" (0.32 pp, larger variance). It is a free trick to enable but should not be load-bearing in claims.
- **LoRA+ ratios are strongly task- and width-dependent.** Hayou et al. report optimal λ ranging 2¹–2⁴ across Llama-7B and RoBERTa-base. For RoBERTa-Large (RoLoRA's setting), λ = 16 is a starting point but should be swept per task; over-aggressive λ destabilizes B-rounds.