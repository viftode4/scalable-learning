# Deep questions for Tuesday 2 June presentations

Group 7 presents on **Tuesday 9 June 2026 at 14:30**, so we must understand the **Tuesday 2 June** papers and prepare one strong critical question per presentation.

Local PDFs are saved in [`required-reading/june-2-papers/`](required-reading/june-2-papers/). Extracted text files are in [`required-reading/june-2-papers/extracted-text/`](required-reading/june-2-papers/extracted-text/).

Use the **bold question** in class. The notes below each question explain why it is deep and what a good answer should mention.

## 13:45 — Group 4 — Federated Residual Low-Rank Adaption of Large Language Models

- **Local PDF:** [`group-04-federated-residual-low-rank-adaption.pdf`](required-reading/june-2-papers/group-04-federated-residual-low-rank-adaption.pdf)
- **Source:** OpenReview PDF, `https://openreview.net/pdf/55447202fb5bd65c71dab03987f3fd0fcf522235.pdf`

### Ask this

**FRLoRA fixes FedAvg+LoRA with two mechanisms at once: it accumulates residual low-rank updates directly into the global model weights, and it reinitializes local LoRA factors in the pretrained model's principal singular space each round. How do the authors disentangle these two effects experimentally, and how can we tell that the improvement comes from mitigating non-IID client drift rather than from gradually giving the global model a higher effective rank or more dense fine-tuning capacity over communication rounds?**

### Why this is a strong question

This question forces the presenters to go beyond "FRLoRA performs better" and explain the mechanism. FRLoRA claims to solve both:

1. **Intrinsic LoRA limitation:** each round's update is rank-constrained.
2. **Extrinsic FL limitation:** non-IID data causes client drift.

But residual accumulation means the final global update may no longer behave like a single low-rank adapter. Over many rounds, it can become a richer accumulated weight change. That may be the real reason performance improves. A good answer should discuss the paper's ablations, rank/capacity interpretation, communication/storage consequences, and whether the method remains parameter-efficient in the same sense as LoRA.

### Follow-up

If the final model is produced by repeatedly adding low-rank residuals into dense weights, what exactly is communicated and stored in a real federated deployment: only LoRA factors, accumulated dense deltas, or a full updated model checkpoint?

---

## 14:00 — Group 20 — Ring Attention with Blockwise Transformers for Near-Infinite Context

- **Local PDF:** [`group-20-ring-attention-blockwise-transformers.pdf`](required-reading/june-2-papers/group-20-ring-attention-blockwise-transformers.pdf)
- **Source:** arXiv PDF, `https://arxiv.org/pdf/2310.01889`

### Ask this

**Ring Attention's core claim is that KV-block communication can be fully hidden under blockwise attention computation, giving near-zero additional overhead and context length scaling with device count. But the paper's own analysis depends on the FLOPS/bandwidth ratio and shows much stricter block-size requirements for slower interconnects such as InfiniBand. Under what hardware, block-size, batch-size, and sequence-length regimes does the overlap assumption fail, and how would you decide between Ring Attention, Ulysses-style all-to-all sequence parallelism, and ordinary memory-efficient attention for a real cluster?**

### Why this is a strong question

This tests whether the presenters understand that Ring Attention is not magic "infinite context"; it is a scheduling trick that works when computation is large enough to hide communication. A shallow presentation will say "communication is overlapped." A good answer should mention:

- ring rotation of KV blocks;
- blockwise exact attention, not approximate attention;
- the condition that block compute time must exceed transfer time;
- why high-bandwidth TPU/NVLink settings are friendlier than slow InfiniBand;
- why low batch size, inference, small blocks, or insufficient arithmetic intensity may expose communication overhead;
- the tradeoff versus Ulysses, which uses all-to-all and is constrained by attention heads but has a different communication pattern.

### Follow-up

The paper frames context length as scaling linearly with device count. What becomes the next bottleneck once activation memory is solved: network bandwidth, latency over many ring steps, optimizer/state memory, positional encoding/generalization, or data quality for extremely long contexts?

---

## 14:15 — Group 14 — Distributed Backdoor Attacks on Federated Graph Learning and Certified Defenses

- **Local PDF:** [`group-14-distributed-backdoor-fedgraph-certified-defenses.pdf`](required-reading/june-2-papers/group-14-distributed-backdoor-fedgraph-certified-defenses.pdf)
- **Source:** arXiv PDF, `https://arxiv.org/pdf/2407.08935`

### Ask this

**The defense certifies robustness by deterministically partitioning a test graph into non-overlapping subgraphs and majority-voting over their predictions, so a bounded trigger can only corrupt a bounded number of votes. But many graph labels depend on global structure, long-range connectivity, or motifs spanning multiple regions. How does the defense trade off certified robustness against loss of global graph information, and in which types of graph-classification tasks would the certificate be formally valid but practically too conservative or accuracy-damaging?**

### Why this is a strong question

This forces them to connect the proof idea to graph semantics. The paper's attack is adaptive and graph-dependent: it learns trigger location and shape, often near important nodes. The defense responds with deterministic partitioning and majority voting. That can certify against arbitrary bounded trigger shape/location, but it may also throw away exactly the global structure that makes graph learning useful.

A good answer should mention:

- Opt-GDBA's adaptive trigger generator and why random graph triggers are weaker;
- the defense's non-overlapping subgraph construction;
- why majority vote gives a deterministic certificate;
- the dependence on trigger-size budget;
- possible clean-accuracy loss when subgraphs are insufficient for the original task;
- whether distributed or spread-out triggers could erode the vote margin faster than localized triggers.

### Follow-up

If an adaptive attacker knows the deterministic partition rule, could they design a low-size trigger that minimally changes each subgraph but flips enough votes across partitions, and how is that covered or not covered by the certificate?

---

## 14:30 — Group 1 — Mitigating Memorization in Language Models

- **Local PDF:** [`group-01-mitigating-memorization-language-models.pdf`](required-reading/june-2-papers/group-01-mitigating-memorization-language-models.pdf)
- **Source:** arXiv PDF, `https://arxiv.org/pdf/2410.02159`

### Ask this

**BalancedSubnet tries to localize and remove weights responsible for memorized artifacts while preserving normal task behavior, using clean/reference data to penalize removal of weights useful for non-memorized generation. In a real production LM, however, we often do not know the full set of memorized strings, and some verbatim recall may be legitimate rare knowledge, quotations, code snippets, or long-tail facts. How does the method distinguish harmful memorization from useful rare knowledge, and how sensitive is that distinction to the choice of clean reference data and memorization benchmark such as TinyMem?**

### Why this is a strong question

This targets the central tension in memorization mitigation: removing private/copyrighted/backdoor artifacts without damaging useful knowledge. The paper defines memorization using elicited artifact sequences and uses TinyMem to make development efficient. BalancedSubnet adds a term meant to avoid pruning weights important for non-memorized sequences. But that relies on the reference data actually representing what should be preserved.

A good answer should mention:

- the paper's `(n, k)` memorization definition;
- TinyMem as a controlled proxy with known injected artifacts;
- the difference between noise artifacts and backdoor artifacts;
- why unlearning can outperform regularization/fine-tuning in their experiments;
- the risk that rare legitimate facts and harmful memorized artifacts are entangled in weights;
- why success on TinyMem may not fully solve auditing in production-scale models.

### Follow-up

If the clean reference data omits a domain, language, or long-tail capability, could BalancedSubnet accidentally classify that capability as safe-to-remove memorization-related circuitry?

---

## 14:45 — Group 19 — Differential Transformer

- **Local PDF:** [`group-19-differential-transformer.pdf`](required-reading/june-2-papers/group-19-differential-transformer.pdf)
- **Source:** arXiv PDF, `https://arxiv.org/pdf/2410.05258`

### Ask this

**Differential Transformer subtracts one softmax attention map from another and interprets the result as canceling common-mode attention noise, which improves retrieval, hallucination, in-context learning robustness, and activation outliers in the paper. But attention to "irrelevant" context is not always noise: in summarization, reasoning, or multi-hop tasks, weak diffuse evidence can matter. What evidence shows that the subtracted attention mass is genuinely harmful common-mode noise rather than useful low-salience signal, and what failure modes would you expect on tasks where evidence is distributed across many tokens instead of concentrated in one answer span?**

### Why this is a strong question

This probes the main conceptual assumption. The method borrows an analogy from differential amplifiers/noise-canceling headphones, but language context is not an electrical signal with clearly separable common-mode noise. A good answer should mention:

- the formula: difference between two softmax maps with learnable lambda;
- the attention sink / irrelevant-context motivation;
- retrieval and hallucination experiments as evidence;
- GroupNorm and lambda initialization/stability details;
- the fact that subtractive attention could introduce negative or suppressed contributions;
- possible failure cases where useful context is broad, redundant, or weakly distributed.

### Follow-up

The ablation shows GroupNorm is important for training stability in DIFF Transformer. Should the reported gain be attributed to the differential attention idea alone, or to a combined architectural package whose stability requirements may complicate scaling and implementation?

---

## 15:00 — Group 11 — Exact Certification of (Graph) Neural Networks Against Label Poisoning

- **Local PDF:** [`group-11-exact-certification-gnn-label-poisoning.pdf`](required-reading/june-2-papers/group-11-exact-certification-gnn-label-poisoning.pdf)
- **Source:** arXiv PDF, `https://arxiv.org/pdf/2412.00537`

### Ask this

**The paper gives exact certificates against label flipping by moving to the NTK/SVM view of sufficiently wide neural networks and solving a MILP for sample-wise or collective robustness. That is mathematically powerful, but practical GNNs are finite-width, trained with finite-time optimizers, regularization, early stopping, and sometimes non-kernel-like dynamics. When we present the guarantee, should we say it certifies the actual deployed GNN, or an infinite-width NTK surrogate of that GNN architecture; and what empirical evidence would convince us that the certificate meaningfully transfers to realistic finite models?**

### Why this is a strong question

This question separates formal exactness from practical applicability. The certificate is exact for the reformulated problem, but the reformulated problem relies on an NTK/SVM equivalence. The MILP is also NP-hard and scales with labeled/test nodes, so the setting matters.

A good answer should mention:

- label flipping changes training labels, not graph structure/features;
- the bilevel poisoning problem is reformulated using NTK/SVM equivalence;
- exact sample-wise vs collective certification;
- high-probability/width assumptions for finite networks;
- MILP complexity: binary variables scale with labeled nodes and test nodes;
- why semi-supervised graph benchmarks with sparse labels are tractable;
- the reported robustness plateau phenomenon and whether it is architecture/data dependent.

### Follow-up

If the method scales best when the number of labeled nodes is small, how would it behave in modern graph learning settings with many labels, inductive sampling, dynamic graphs, or large heterogeneous graphs?

---

## 15:15 — Group 3 — On the Byzantine-Resilience of Distillation-Based Federated Learning

- **Local PDF:** [`group-03-byzantine-resilience-distillation-fl.pdf`](required-reading/june-2-papers/group-03-byzantine-resilience-distillation-fl.pdf)
- **Source:** arXiv PDF, `https://arxiv.org/pdf/2402.12265`

### Ask this

**The paper argues that distillation-based FL is naturally more Byzantine-resilient because malicious clients can only send predictions on a public dataset, so their attack space is bounded by the probability simplex and influences the server indirectly through the distillation objective. But this also makes the public/proxy dataset the entire communication bottleneck between private client data and the server. How do the attacks and defenses change if the public dataset is distribution-shifted, too small, class-imbalanced, or partially controlled by the adversary; and does prediction-space robustness still imply useful robustness for the private client distributions we actually care about?**

### Why this is a strong question

This targets the key assumption behind FedDistill. The paper's resilience story depends on replacing parameter updates with predictions on public data. That constrains Byzantine clients, but it also means all robustness and learning signals pass through that public dataset.

A good answer should mention:

- why FedDistill's attack surface is lower-dimensional than FedAvg's parameter space;
- why Byzantine clients influence the server indirectly through labels/logits rather than direct weights;
- LMA/CPA as FedDistill-specific attacks;
- ExpGuard's use of historical and current client behavior;
- the assumption that honest private data and public data come from the same distribution;
- why non-IID honest clients can look suspicious under history-based defenses;
- HIPS as an obfuscation method that makes attacks harder to detect.

### Follow-up

ExpGuard relies on persistent client identities and historical behavior. What happens if clients are ephemeral, participate intermittently, or honestly shift over time, as in realistic cross-device FL?
