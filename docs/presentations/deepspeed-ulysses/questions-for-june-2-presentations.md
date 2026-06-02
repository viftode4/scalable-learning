# Critical questions — Tuesday 2 June 2026 presentations

Group 7 (us) presents **Tuesday 9 June 2026, 14:30**. For the **2 June** session we each ask one strong, critical question per assigned paper.

**Professor's bar:** *"A good question goes beyond simple clarification and engages critically with the paper's methodology, results, or broader implications."*

So every question below names a specific thing in the paper, then puts pressure on it — none are "can you explain X."

**Ownership**
- **Vlad:** Group 1, Group 19, Group 11, Group 3
- **Daniel:** Group 4, Group 20, Group 14

Local PDFs: [`required-reading/june-2-papers/`](required-reading/june-2-papers/) · extracted text: [`required-reading/june-2-papers/extracted-text/`](required-reading/june-2-papers/extracted-text/).

All premises below were verified against the papers. Where a question was rejected for being *already answered in the paper*, that is noted so we don't ask it.

---

## 13:45 — Group 4 — Federated Residual Low-Rank Adaption (FRLoRA) · *Daniel*

[`group-04-federated-residual-low-rank-adaption.pdf`](required-reading/june-2-papers/group-04-federated-residual-low-rank-adaption.pdf) · arXiv via OpenReview

### Ask
**"FRLoRA accumulates low-rank residuals into the global weights every round, so over many rounds the effective update is no longer a single low-rank adapter — its rank can grow up to r(T+1). How do you show the gain comes from mitigating non-IID client drift, rather than simply from giving the global model more effective capacity over time? And in a real deployment, what is actually communicated and stored — only LoRA factors, or accumulated dense deltas?"**

- **Engages with:** methodology (disentangling two coupled mechanisms) + implications (communication/storage cost).
- **Dodge tell:** "FRLoRA just performs better" without separating drift-mitigation from accumulated rank/capacity.
- **Note for asker:** the principal-singular-space init is computed *once* from the pretrained weights and reused each round — not re-SVD'd per round (their per-round-SVD variant performed *worse* due to SVD instability). Don't claim it re-decomposes each round.

---

## 14:00 — Group 20 — Ring Attention with Blockwise Transformers · *Daniel*

[`group-20-ring-attention-blockwise-transformers.pdf`](required-reading/june-2-papers/group-20-ring-attention-blockwise-transformers.pdf) · arXiv 2310.01889

### Ask
**"Ring Attention hides KV-block communication under blockwise compute only when block compute time exceeds transfer time — the paper's own condition is block size ≥ FLOPS/bandwidth, and InfiniBand needs ~24× larger blocks than NVLink. So under what batch-size, sequence-length, and interconnect regimes does the 'zero-overhead' claim actually break, and how would you choose between Ring Attention and Ulysses-style all-to-all for a real cluster?"**

- **Engages with:** results (the overlap guarantee is conditional, not free) + implications (hardware-dependent design choice).
- **Dodge tell:** "communication is fully overlapped" with no mention of the compute > transfer condition or low-arithmetic-intensity regimes (small batch, inference, small blocks).
- **Follow-up:** once activation memory is solved, what becomes the next bottleneck — latency over many ring steps, optimizer-state memory, or long-context generalization?

---

## 14:15 — Group 14 — Distributed Backdoor on FedGraph + Certified Defense · *Daniel*

[`group-14-distributed-backdoor-fedgraph-certified-defenses.pdf`](required-reading/june-2-papers/group-14-distributed-backdoor-fedgraph-certified-defenses.pdf) · arXiv 2407.08935

### Ask
**"The defense partitions a test graph into non-overlapping subgraphs and majority-votes, so a bounded trigger corrupts a bounded number of votes. But many graph labels depend on global structure spanning multiple regions — and you report a large certified-vs-clean accuracy gap on dense datasets like PROTEINS. On which tasks is the certificate formally valid but practically too conservative, and could a distributed trigger erode the vote margin faster than a localized one?"**

- **Engages with:** methodology (the proof discards the global structure that makes graph learning useful) + results (the certified-accuracy gap).
- **Dodge tell:** calling the 40–60% certified accuracy on dense graphs "acceptable" without discussing the utility trade-off, or not noting m* depends on knowing the trigger-size budget.
- **Note for asker:** the adaptive attack is **Opt-GDBA** (learns trigger location/shape, attaches to central nodes); random-trigger Rand-GDBA is the weaker baseline.

---

## 14:30 — Group 1 — Mitigating Memorization in Language Models · *Vlad*

[`group-01-mitigating-memorization-language-models.pdf`](required-reading/june-2-papers/group-01-mitigating-memorization-language-models.pdf) · arXiv 2410.02159

### Ask
**"BalancedSubnet doesn't *find* memorized content — you hand it the list of strings to delete, and the retain set just protects everything else. In production no one has that clean list, and some verbatim recall is legitimate: rare facts, quotations, code. Where is that target list supposed to come from without sweeping in legitimate rare knowledge, and if your retain set omits a domain, do those weights just look 'safe to remove'?"**

- **Engages with:** methodology (the method presupposes a detection step it does not provide) + implications (production auditing).
- **Dodge tell:** "we preserve perplexity while removing memorization" — true only for what the retain set covers; or implying the method discovers memorized content itself.
- **Grounding:** the `(n,k)` definition is purely mechanical (greedy decoding reproduces the continuation), so it can't separate a memorized SSN from a famous quotation — which is why they inject artifacts they already control (TinyMem). The "works on production models" claim only tests on sequences someone else pre-extracted.

---

## 14:45 — Group 19 — Differential Transformer · *Vlad*

[`group-19-differential-transformer.pdf`](required-reading/june-2-papers/group-19-differential-transformer.pdf) · arXiv 2410.05258

### Ask
**"Your evidence that the subtracted attention is 'noise' is downstream — retrieval and hallucination scores go up. But higher end-task accuracy doesn't show the discarded attention mass was useless rather than low-salience-but-useful, and the subtraction can go net-negative on a token. Is there any direct measurement of what's being thrown away, and a case where genuinely relevant but diffuse evidence gets suppressed?"**

- **Engages with:** results (whether the metrics actually support the "noise-cancelling" claim).
- **Say "attention noise," not "attention sink"** — the paper never uses "attention sink."
- **Dodge tell:** citing the retrieval numbers or the noise-cancelling-headphones analogy as if they settle it; note they *already* test distributed-evidence tasks (summarization, multi-hop QA) and win — so press that wins don't prove no useful signal was discarded.
- **Follow-up (NOT answered in the paper):** "λ is learnable and per-layer, but you only show robustness to its *initialization*. Did you analyze what λ actually converges to across layers and training — is the benefit mild denoising, or the model learning to mostly discard the second map?"
- **Rejected follow-up:** "is the gain differential attention or the GroupNorm package?" — Section 3.8 / Table 6 already runs that ablation (GroupNorm on vanilla does nothing; removing it from DIFF hurts via instability). Do **not** ask this; they'll point to the ablation.

---

## 15:00 — Group 11 — Exact Certification of GNNs Against Label Poisoning · *Vlad*

[`group-11-exact-certification-gnn-label-poisoning.pdf`](required-reading/june-2-papers/group-11-exact-certification-gnn-label-poisoning.pdf) · arXiv 2412.00537

### Ask
**"The certificate is exact — but exact for the infinite-width NTK/SVM idealization of the GNN; for finite width you only have a high-probability bound, and every experiment is run in that idealized limit. When you report this guarantee, are you certifying the deployed network or its idealization — and is there any finite-width experiment showing the certificate actually transfers?"**

- **Engages with:** methodology + implications (formal exactness vs. the model people actually deploy).
- **Dodge tell:** treating the `O(ln w/√w)` high-probability bound as if it were empirical validation on real finite networks.
- **Do NOT ask about scaling / dynamic / inductive graphs** — App. H.2.7 (more labeled nodes), H.4 (dynamic/inductive), and the Cora-ML/Citeseer ceiling already cover that, and the inexact multi-class relaxation (Table 2) is their scaling escape hatch.

---

## 15:15 — Group 3 — Byzantine-Resilience of Distillation-Based FL · *Vlad*

[`group-03-byzantine-resilience-distillation-fl.pdf`](required-reading/june-2-papers/group-03-byzantine-resilience-distillation-fl.pdf) · arXiv 2402.12265

### Ask
**"Your resilience story rests on the public dataset matching the clients' private distribution — which makes that one dataset the channel everything passes through. When it's distribution-shifted, small, or class-imbalanced — or partly adversary-chosen — do LMA/CPA and ExpGuard still hold, and does robustness measured in prediction space still buy robustness for the private distributions we actually care about?"**

- **Engages with:** assumptions + implications (a load-bearing assumption tested only under IID).
- **Live ammo:** they reuse CIFAR-100's train split as the public set and admit it "does not reflect a realistic use-case" — a concrete instance of exactly this problem.
- **Dodge tell:** citing the `√2·α·C` gradient-error bound as if it were distribution-free — both the constant C and the honest target shift under distribution mismatch.
- **Follow-up (strongest item — keep it):** "ExpGuard accumulates trust over rounds and needs persistent client identities. In cross-device FL clients are ephemeral and honest clients drift over time — does the history-based weighting still work, or do honest-but-changing clients start looking Byzantine?"

---

### How to score engagement on the day
Across all four, the clearest sign a presenter understands their own paper's limits is whether they **volunteer the concession** — the untested transfer, the missing measurement, the out-of-model assumption — before we have to pull it out of them. If they only recite the abstract or point to a result that doesn't address the gap, that's the dodge.
