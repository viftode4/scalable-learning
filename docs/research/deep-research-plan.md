# Project Plan & Technical Decision Document
**TU Delft CS 4725 — Reproducing & Extending RoLoRA (Chen et al., NeurIPS 2025)**
*3 people · 10 weeks · DelftBlue + DAIC*

---

## TL;DR
- **Reproduction is feasible but tight.** The headline RoBERTa-Large + 50-client MNLI/QQP/QNLI result is roughly a **30–60 GPU-hour** job per (method × dataset × seed) on a V100/A40 in a sequential single-GPU simulator. With 3 baselines (LoRA, FFA-LoRA, RoLoRA) × 3 datasets × 3 client counts × 3 seeds you should budget **~600–900 GPU-hours total** for the headline table; this is achievable on DelftBlue's `gpu-a100` and `gpu-v100` partitions across a 10-week window if you start scripting in week 1. The Llama-2-7B stretch is a separate 200–400 GPU-hour commitment and should remain optional. Use `gpu-a100-small` (10 GB MIG slices, fast queue) for all RoBERTa work, `gpu-a100` (full 80 GB) only for Llama.
- **Official RoLoRA federated code is not publicly released on GitHub as of May 2026.** It exists only inside the NeurIPS 2025 OpenReview supplementary zip (forum `u4mobiHTJl`); neither author (Shuangyi Chen, Yuanxin Guo) has a public RoLoRA repo. The realistic path is: **fork FedSA-LoRA's repo (Pengxin-Guo/FedSA-LoRA) as your baseline harness** because it already implements LoRA + FFA-LoRA + FedSA-LoRA on RoBERTa with FedAvg infrastructure; add RoLoRA's odd/even alternation in ~50 LOC. Do **not** invest in FederatedScope-LLM as your main vehicle — it is heavy, has rough edges, and its LLM branch is older than the FedSA-LoRA harness. Cross-reference Fed-SB's repo (CERT-Lab/fed-sb), which contains a reimplementation of RoLoRA you can sanity-check against.
- **Best improvement angle for grade 10 + workshop potential is (a)+(g) combined: "Robust RoLoRA under partial participation with communication-time-aware scheduling."** This is the angle the paper *explicitly punts on* in Appendix A2, has not been claimed by the November 2025 / February 2026 follow-ups (ADF-LoRA, FedRot-LoRA, FedMomentum, LA-LoRA, SDFLoRA), is empirically tractable on RoBERTa-Large, and naturally produces a paper-shaped story (analysis of stale-A vs stale-B, then a fix). Angles (c) LoRA+ asymmetric LR and (d) warmup-then-freeze are excellent **secondary** experiments that fit into the same codebase and round out the report. Angles (e), (f) are crowded; (b) is interesting but FedSA-LoRA already partially occupies it.

---

## Key Findings

### 1. Code reality check (the most important practical fact)
- **Official RoLoRA code: not on public GitHub.** Searches of dblp/Khisti's group page, Shuangyi Chen's and Yuanxin Guo's personal pages (guoyuanxinkevin.github.io confirms Yuanxin Guo is a Toronto PhD, no repos linked to RoLoRA), and GitHub's name search return nothing under `RoLoRA` that is the federated paper. There is a totally distinct **HuangOwen/RoLoRA** (EMNLP 2024 *quantization* paper) that you must **not** confuse — it's a different paper, different team. The OpenReview entry `u4mobiHTJl` exists with a "Code" supplementary, which means the team can download the supplementary zip from OpenReview after logging in; that is the only authoritative source. Plan as if it might be a partial / messy research-grade dump (this is typical for theory-heavy NeurIPS submissions whose code is in the supplementary rather than a polished GitHub repo).
- **Recommended harness: fork `Pengxin-Guo/FedSA-LoRA`.** Public, 51 stars, ICLR 2025, Python 3.10 + PyTorch 2.1, already implements `LoRA`, `FFA-LoRA`, `FedSA-LoRA` on RoBERTa-base/large for GLUE under FedAvg with Dirichlet-α non-IID splits. Adding RoLoRA = one boolean per round flipping `requires_grad` on A vs B and changing what each client uploads. ~50 lines.
- **Cross-check against `CERT-Lab/fed-sb`.** Fed-SB explicitly benchmarks against RoLoRA, so its repo contains a third-party RoLoRA implementation usable for numerical sanity-checking your fork.
- **Newer relevant repos worth knowing:** `CERT-Lab/fedex-lora` (FedEx-LoRA, ACL'25 oral, exact aggregation via residual matrix); `alibaba/FederatedScope/tree/llm` and `tree/FlexLoRA` (heavy; only useful if you need rank-heterogeneity baselines).

### 2. Compute reality on DelftBlue
DelftBlue partitions you will actually use (per official docs):
- `gpu-a100-small` — MIG slice, 10 GB VRAM, ≤4 h jobs, ≤1 GPU. **Fastest queue.** 10 GB is enough for RoBERTa-Large LoRA (≤6 GB at fp16/bs=16) — this should be your default.
- `gpu-v100` — 4× V100 32 GB nodes (Phase 1).
- `gpu-a100` — 4× A100 80 GB nodes (Phase 2). Required for Llama-2-7B (full-precision LoRA needs ~28 GB; even with 4-bit quantization plan ~14–18 GB).
- Default jobs run at low priority. **Get your supervisor (Atasu/Chen) to put you on a faculty share via TOPdesk in week 1** — without it, your jobs may queue 8–24 h on busy days. DAIC is your backup; INSY can be useful for 4-h iteration jobs.

### 3. Compute estimates (per reproduction target)
Anchored on the paper's reported hardware (RTX 4090 / A40, ~50% slower than A100 for fp16 LoRA training but similar to V100). The paper trains 50-client RoBERTa-Large with rank 4 for hundreds of communication rounds in a single-GPU **simulator** (clients run sequentially), which is the dominant cost driver.

| Target | Setup | Per-run wall-clock on 1×A100 (single-GPU sim) | GPU-hours per cell |
|---|---|---|---|
| MNIST 2-layer toy (Fig. 2) | CPU/laptop, rank 1, ~30 s/run | 1 min | <0.1 |
| RoBERTa-Large MNLI, 3 clients, rank 4, ~30 rounds × 20 local epochs, bs=32 | full dataset, 3 clients ⇒ each client sees ~130k examples per round-epoch | ~2–4 h on A100, ~4–6 h on V100 | 3 |
| RoBERTa-Large MNLI, **20 clients**, rank 4 | each client sees ~20k examples; sequential client loop | ~3–6 h | 5 |
| RoBERTa-Large MNLI, **50 clients**, rank 4 | each client sees ~8k examples but loop overhead dominates | ~5–10 h | 8 |
| Same for QQP (~360k train) | similar | ~8 h | 8 |
| Same for QNLI (~105k train) | smaller | ~3 h | 3 |
| Local-steps ablation (Table 4): 3 clients × 3 step settings × 1 dataset | | ~12 h | 12 |
| **Llama-2-7B commonsense, 50 clients, rank 8 (stretch)** | data smaller (~10–20k) but 7B model is the bottleneck; ~30 rounds | 24–48 h on 1×A100-80GB | 36 |

**Headline reproduction budget (3 baselines × 3 datasets × 3 client counts × 3 seeds with shared dataset preprocessing):**
- 27 cells × ~5 GPU-h average ≈ **135 GPU-h** for the headline + duplication/seeds factor 2× for variance ⇒ **~270 GPU-h on A100-equivalent**, or **~400–500 GPU-h on V100-class**, or **~600–900 V100-h with queue overhead, debugging reruns, and seed-3 redundancy** included.
- Local-steps ablation adds **~50 GPU-h**.
- Llama stretch adds **~200–400 GPU-h** if you include it.

**Realistic wall-clock with queueing on a shared cluster:** if you submit ~12 jobs concurrently to `gpu-a100-small` (which is fast-queue) you can chew through ~50 GPU-h/day across the team. Headline reproduction therefore takes **~6–10 calendar days of compute** if scripts work first time, and **2–3 calendar weeks** in practice. This is why setup must finish by week 3.

### 4. Improvement angles — ranked evaluation

Scoring rubric (1–5): N=novelty, T=tractability in 10 wks/3 ppl, P=likelihood of positive empirical signal, W=workshop-paper potential.

| # | Angle | N | T | P | W | Notes |
|---|---|---|---|---|---|---|
| **(a)+(g)** | **Partial participation + comm-time-aware alternation** | **4** | **4** | **4** | **5** | **Recommended primary.** Paper explicitly punts in Appendix A2; no follow-up fully addresses it. Stragglers expose a structural asymmetry: stale A is theoretically more harmful than stale B (down-projection determines the subspace; B is a linear head over fixed A). Ample story: characterize, then propose "skip A-rounds when stragglers exceed threshold" or "B-only catch-up rounds for stragglers." Plus you can naturally include the comm-time analysis (Figure 12 only does 3 clients; redo at 50). |
| (c) | LoRA+ asymmetric LR sweep beyond 4× | 3 | 5 | 3 | 3 | Easy add-on. Hayou et al. ICML'24 recommend 16× and show it depends on width — RoBERTa-Large width 1024 is in the regime where 8–32× should help. Paper only tested 2×/4×. **Use as a secondary experiment**: one figure plus one table, low risk. |
| (d) | Warmup-then-freeze (5–20% RoLoRA → FFA-LoRA) | 3 | 5 | 4 | 3 | Cheap, intuitive, paper only tested coarse mixes. Likely positive: matches LoRA-init-then-freeze folklore. Combine with (c) for the report. |
| (b) | RoLoRA × FedSA-LoRA partial-aggregation | 3 | 3 | 3 | 3 | Conceptually cute but FedSA-LoRA's "share-A" already disagrees with RoLoRA's "share whatever isn't frozen this round". Hybrid is "share only A on A-rounds, do not share B" — basically equivalent to FedSA on alternating epochs. Risk of being a no-op or marginal. |
| (e) | Heterogeneous-rank clients | 2 | 3 | 3 | 1 | **Crowded.** FlexLoRA, FLoRA, LoRA-A², HetLoRA, FediLoRA, FLoRG all attack this. Avoid. |
| (f) | DP-RoLoRA properly | 4 | 2 | 3 | 2 | **LA-LoRA (Liu et al., arXiv 2602.19926, "Rethinking LoRA for Privacy-Preserving Federated Learning") already crowds this exact niche** — it explicitly analyzes RoLoRA's noise asymmetry, proposes local (within-round) alternation, and outperforms RoLoRA by 16.83% on Tiny-ImageNet at ε=1. Hard to do something novel here in 10 weeks; budget risk too. |

**Fresh angles from literature monitoring (late 2025–early 2026), confirming the partial-participation gap:**
- **ADF-LoRA** (Wang et al., arXiv 2511.18291, Nov 2025): extends alternating LoRA to **decentralized** FL; identifies "phase-state mismatch" and "block-wise divergence" — these are exactly the kinds of failure modes you'll see under partial participation in *centralized* RoLoRA. Mine their analysis but don't duplicate.
- **FedRot-LoRA** (arXiv 2602.23638, Feb 2026): "rotational misalignment" of factor-wise averaging. Different angle, doesn't conflict with yours.
- **FedMomentum** (arXiv 2603.08014, Mar 2026): explicitly criticizes RoLoRA for "zig-zag optimization dynamics" hindering momentum. Their critique matters: when your alternation period interacts with optimizer state, things break. **Worth measuring in your local-steps ablation.**
- **SFed-LoRA** (arXiv 2603.08058, Mar 2026): rsLoRA-style scaling factor for federated rank stability. Tangential.
- **LA-LoRA** (arXiv 2602.19926): claims local-within-round alternation > RoLoRA's between-round alternation under DP. If you have time, run *non-DP* LA-LoRA-style local alternation as a new baseline.
- **FLoRG** (arXiv 2602.17095): Procrustes alignment + low-rank Gram matrices. Tangential.
- **SDFLoRA** (arXiv 2601.11219): selective decoupled federated LoRA with heterogeneity. Tangential.
- **AltLoRA** (centralized, alternating): not a federated method but the alternating-projection idea is a natural connection.

The cleanest niche left is the partial-participation question + its interaction with the alternation cycle — and that is the gap the paper itself flags.

### 5. Risk analysis

**Probability the reproduction itself fails to match published numbers (within ±2% absolute):**
- 3-client headline (Table 1, top rows): ~80% match expected. These are easy.
- 50-client MNLI cliff (LoRA collapses to ~52%, RoLoRA holds ~83%): **~60% match expected.** This number is the headline. It depends sensitively on (i) data partition seed (the paper used non-overlapping partitions of MNLI; reproducing the *exact* partition requires their data split script which may or may not be in the supplementary), (ii) learning rates (paper sweeps; if you copy default you may underperform LoRA further or RoLoRA less), (iii) number of communication rounds — paper does many rounds, time-limited reruns truncate. Plan two seeds × LR sweep over {1e-4, 3e-4, 1e-3, 3e-3} for the LoRA baseline alone. Even if your absolute numbers shift, the *qualitative gap* (LoRA collapses, RoLoRA holds) is robust per multiple independent confirmations (FedSA-LoRA paper, Fed-SB paper, LA-LoRA paper).
- Local-steps ablation (Table 4): ~75% match expected.
- Llama-2-7B commonsense (Table 2/3): ~50% match expected within ±2%, ~80% within ±5%. Llama runs are noisy and reproducibility on commonsense reasoning is folkloric: report deltas to your own LoRA baseline rather than absolute numbers.

**Probability the chosen improvement is neutral or negative:**
- Partial participation: ~25% the alternation-aware fix turns out to be a wash (i.e., naive RoLoRA already works). Even then you have a **negative result + analysis** = still a strong report (this is what the paper itself avoided doing). Mitigation: design experiments so the *characterization* (stale-A vs stale-B asymmetry) is the contribution, and "the fix doesn't help" is acceptable.
- LoRA+ at higher asymmetry: ~40% it doesn't help RoLoRA specifically (because A and B are alternated, the LoRA+ argument about *simultaneous* updates partially breaks down). Useful negative result either way.
- Warmup-then-freeze: ~30% it doesn't beat full RoLoRA but ~80% it beats FFA-LoRA. Safe.

**Common pitfalls in federated LoRA implementations (from reading the literature carefully):**
1. **Aggregation bug 1 — averaging products.** Some codebases (notably early FederatedScope-LLM forks) accidentally compute `avg(A·B)` server-side after merging; this is what RoLoRA exists to fix and what FedAvg-of-LoRA does *not* do. Make sure your LoRA baseline aggregates A and B *separately* (this is what makes the avg(AB)≠avg(A)·avg(B) bug appear in the first place).
2. **LoRA-α / scaling discrepancy.** PEFT library uses `alpha/r`, some other implementations use `alpha/sqrt(r)` (rsLoRA). Cross-implementation comparisons can shift accuracy 2–5% silently. Lock in one convention and verify.
3. **Initialization mismatch.** Several papers (Hayou et al., Init[A] vs Init[B]) report A non-zero / B zero. FFA-LoRA *requires* this convention. If you accidentally make B non-zero, FFA-LoRA's randomness amplifies. Verify with a unit test.
4. **Eval split bug.** GLUE has dev/test — many federated papers report dev (test labels are hidden). Be explicit and consistent. Multiple federated LoRA papers conflict on numbers partly because of this.
5. **Local epoch vs local step semantics.** Paper says "20 local epochs"; some baselines mean "20 local SGD steps". If the FedSA-LoRA harness uses local steps and you reproduce the paper assuming epochs, you can be off by 100×.
6. **Client subsampling order across seeds.** When you set seed=0 for 50-client partitioning, make sure each method sees **the same partition** under the same seed. Seed leak across methods (each method gets its own draw) inflates variance and confuses cross-method comparisons.
7. **Optimizer state at server.** RoLoRA freezes A in even rounds; do you persist Adam moments for the frozen tensor or zero them? The paper is silent. Choose and document; FedMomentum's critique flags this.
8. **Tokenizer cache / dataset preprocessing.** Re-tokenizing GLUE on every job is a 30-minute waste. Cache once on `/scratch`.

### 6. Concrete week-by-week plan

| Wk | Milestones | Person 1 (Infra) | Person 2 (Algorithm) | Person 3 (Improvement/Analysis) | Compute target |
|----|----|----|----|----|---|
| 1 | Repos, accounts, OpenReview supp download | TOPdesk request (faculty share); set up DelftBlue/DAIC env; clone FedSA-LoRA + Fed-SB; download RoLoRA OpenReview supplement | Read both RoLoRA arXiv versions in detail; reimplement MNIST Figure 2 on laptop | Lit-review pass: ADF-LoRA, FedMomentum, FedRot-LoRA, LA-LoRA, FedSA-LoRA, Fed-SB, FFA-LoRA, LoRA+ | laptop only |
| 2 | LoRA baseline reproduces FedSA-LoRA's published numbers on MNLI 3-client | dataset prep script + caching; Slurm job templates; unit tests for aggregation math | port FedSA-LoRA harness; run LoRA baseline 3-client MNLI; verify ±1% of paper | sketch partial-participation simulator hooks (random client mask each round) | ~20 GPU-h |
| 3 | RoLoRA + FFA-LoRA implemented and matched on 3-client | log/W&B integration | implement RoLoRA odd/even alternation; verify FFA-LoRA matches paper Table 1 row 2; verify RoLoRA matches paper Table 1 row 3; run sanity at rank 1, 4, 8 | implement LoRA+ asymmetric-LR option in same codebase | ~40 GPU-h |
| **4** | **PROPOSAL DUE** — submit proposal: "Reproduce RoLoRA + investigate partial-participation robustness" | freeze the harness; write infra section of proposal | write algorithm section + reproduction plan; provide preliminary 3-client numbers as evidence | write motivation: identify gap (Appendix A2 of paper), formalize stale-A vs stale-B asymmetry hypothesis | — |
| 4–5 | **Homework weeks** — split bandwidth | finish 20-client + 50-client MNLI reproduction | start QQP/QNLI 3-client | first partial-participation experiments (random 50% client dropout per round, 20-client MNLI) | ~80 GPU-h |
| **6** | **MIDTERM REVIEW** — present reproduction status + 1 preliminary improvement plot | wrap 50-client MNLI/QQP/QNLI | local-steps ablation (Table 4) | partial-participation extended: stale-A vs stale-B isolation experiment | ~80 GPU-h |
| 7 | **PAPER PRESENTATION** (course slot) | run remaining 50-client cells; consolidate plots | LoRA+ asymmetric-LR sweep at 2×/4×/8×/16×/32× | warmup-then-freeze (5%, 10%, 20% mix); start writing analysis section | ~60 GPU-h |
| **8** | **DRAFT REPORT DUE** | clean code release + reproducibility appendix | final ablation runs; commsense Llama-2-7B if time and quota allow | write improvement section with both descriptive (characterization) and prescriptive (proposed fix) parts | ~60 GPU-h |
| 9 | **FINAL REPORT + PRESENTATION** | rerun anything that didn't seed-converge; final figures | review all numbers vs paper Table 1; compile reproduction-success report | finalize claims; ensure each claim has ≥3 seeds | ~30 GPU-h |
| 10 | Buffer / workshop submission polish | reproducibility README, scripts, cluster env | write workshop-paper version (FL@FM workshop, ES-FoMo cycle) | refine analysis + theory hooks | — |

**Kill criteria (pivot points):**
- **End of week 2** — if LoRA baseline doesn't reproduce within ±2% on 3-client MNLI: stop trying to reimplement; switch to running Fed-SB's repo verbatim (it has all baselines).
- **End of week 3** — if RoLoRA doesn't reproduce within ±2% on 3-client: contact authors via Khisti email (akhisti@ece.utoronto.ca, listed publicly), they're responsive based on dblp activity. **Set a 1-week deadline on author response.**
- **End of week 5** — if 50-client MNLI cliff not reproduced (LoRA collapses, RoLoRA holds): pivot to using LoRA-A² or FedSA-LoRA's own evidence of the cliff (both papers reproduce it independently), or shift the project framing from "reproduce the cliff" to "characterize the cliff under realistic FL conditions."
- **End of week 7** — if partial-participation experiment shows no signal at all: pivot improvement story to LoRA+ × RoLoRA combination (angles c+d) which is much safer.

### 7. Role split — validation

Splitting by **ownership-layer** (infra / algorithm / improvement) is **better than splitting by experiment**, with one caveat:
- ✅ **Pros confirmed:** each layer has clear interfaces; one Slurm-fluent person, one PyTorch/PEFT person, one writing/analysis person who also drives the novelty. Cross-coverage on each PR (every commit reviewed by at least one other) gives all three people genuine breadth — important for individual grading.
- ⚠️ **Caveat:** the "infra" person can become a queue-bottleneck if the cluster is flaky in weeks 5–7. Mitigation: weeks 5–6 the algorithm/improvement people both should be cleared to submit jobs, with the infra person owning only environment setup, caching, and reproducibility scripts.
- ✅ **Each person writes code AND runs experiments AND writes** — keep this. Specifically, give each person ownership of one section of the final report (Reproduction / Improvement / Analysis) and one section of the appendix. This protects you against the failure mode where one teammate ends up only "running scripts."

Critique of the alternative ("split by experiment"): it tightly couples each person to one dataset/setting, which means if MNLI fails you lose a teammate's work. The layered split is more robust. **Use the layered split.**

### 8. Hardware-tier ladder (cheapest meaningful experiment first)

| Tier | What you can run | Cheapest meaningful claim |
|---|---|---|
| Laptop CPU | MNIST 2-layer toy (Figure 2 of paper) at rank 1 | "We reproduce the paper's central theoretical claim that FFA-LoRA's fixed A is a *systematic* (not stochastic) bottleneck — alternating updates close the gap on the toy 2-layer model." (~5 minutes) |
| Laptop GPU (RTX 3060 / M-series) | RoBERTa-Large rank-1 LoRA on MRPC, 3 clients, 5 rounds | "Aggregation math is correctly implemented: avg(A·B) ≠ avg(A)·avg(B), and we measure the residual error per round as in Figure 3." Takes ~30 minutes. |
| 1×V100, ~6 h job | RoBERTa-Large MNLI 3-client all 3 baselines, 1 seed | "We reproduce paper Table 1 top block." This validates the harness end-to-end. |
| 1×V100, ~24 h job (or 4 jobs in parallel) | RoBERTa-Large 20-client MNLI/QQP/QNLI all 3 baselines, 2 seeds | "We reproduce the scaling-with-client-count trend." |
| 1×A100, ~2 days × 12 jobs (parallel) | Full headline Table 1, 50-client × 3 datasets × 3 baselines × 3 seeds | "Full reproduction." |
| 1×A100 80GB, ~3 days | Llama-2-7B commonsense 50-client | Stretch, only if quota permits. |
| Multi-GPU (≥2×A100) | Not actually needed here — single-GPU simulator is fine. Only useful if you go to Llama-2-13B (do not). | — |

**Hard rule:** every experiment should be runnable end-to-end in ≤4 h to fit in `gpu-a100-small`. If a single seed × cell needs more, restructure (checkpoint + resume).

### 9. Surprising things to know (federated LoRA literature gotchas)

- **The same name "RoLoRA" refers to two unrelated papers.** Yours: Chen et al. (Toronto/Ericsson, federated, alternating LoRA, NeurIPS 2025). Other: Huang et al. (EMNLP 2024, *quantization-aware* LoRA, github.com/HuangOwen/RoLoRA). Several lit-search tools blur them. Always cite arXiv ID.
- **FFA-LoRA's own paper acknowledges alternating updates and dismisses them.** From Sun et al. ICLR 2024: *"Another intuitive approach … is to alternatively update the two LoRA weights. While this update method exhibits similar properties, it is empirically shown to be slow to converge."* RoLoRA's contribution is partly to disprove this offhand claim. Keep this in mind: **don't be surprised if your reproduction finds RoLoRA slower-per-round than FFA-LoRA but better in final accuracy.** Report both axes.
- **Concurrent publication landscape is dense.** RoLoRA (Sep 2024 workshop → Feb 2025 arXiv → NeurIPS 2025), LoRA-A² (Oct 2024, ACL 2025), FedSA-LoRA (Oct 2024, ICLR 2025), Fed-SB (Feb 2025), FedEx-LoRA (Oct 2024, ACL 2025). All claim improvements over each other; cross-comparison numbers are inconsistent across papers. **Don't trust borrowed numbers; rerun any baseline you cite.**
- **FedMomentum (Mar 2026) explicitly criticizes RoLoRA's "zig-zag dynamics."** Their argument: alternation breaks Adam momentum accumulation. This is a real concern for your local-steps ablation — try resetting vs persisting optimizer state when blocks switch.
- **LA-LoRA (Feb 2026) generalizes RoLoRA to within-round alternation and reports that RoLoRA loses 16.83% to LA-LoRA on Swin-B Tiny-ImageNet at ε=1.** This means the DP-RoLoRA angle (f) is not just crowded — RoLoRA is now beaten there. Avoid that comparison in your improvement direction.
- **The 50-client MNLI cliff for vanilla LoRA is reproduced in at least three independent papers** (RoLoRA, FedSA-LoRA, LA-LoRA). High confidence the qualitative effect is real.
- **FedSA-LoRA's argument ("A is general, B is client-specific") and RoLoRA's argument ("A determines the subspace, freezing it is harmful") sound contradictory but aren't:** FedSA-LoRA *aggregates* A and keeps B local; FFA-LoRA *freezes* A; RoLoRA alternates. Three different design choices on what to do with A. The team should be ready to articulate this taxonomy in the final report (it's also a natural framing for the introduction).
- **PEFT library defaults change across versions.** `peft==0.7` vs `peft==0.10` differ in how `lora_alpha` interacts with `r`, and one update silently changed Init[B]. Pin your version.
- **FederatedScope-LLM is a research-grade codebase, not a hardened framework.** Multiple users report difficult installs, CUDA pinning issues, and broken evaluation scripts on certain dataset combinations. Avoid as your primary harness.
- **Khisti's group's prior preprint (the ICML 2024 ES-FoMo workshop version of RoLoRA, arXiv 2409.02346)** is shorter and less polished than the NeurIPS version but contains useful intuitions you can cite for the introduction. Both versions are public.
- **DelftBlue's `gpu-a100-small` MIG slice (10 GB) is enough for RoBERTa-Large LoRA but NOT for Llama-2-7B even quantized.** Plan accordingly: small partition for the bulk of your work, full A100 only for stretch.

---

## Details — the recommended improvement direction in full

**Working title:** *"Robust Federated LoRA Alternation under Partial Client Participation"*

**Concrete experimental plan:**

1. **Characterization (week 5–6):** With 50 clients and a configurable participation rate p∈{0.2, 0.5, 0.8, 1.0}, measure on RoBERTa-Large MNLI:
 - LoRA, FFA-LoRA, RoLoRA accuracy at each p
 - Decompose RoLoRA accuracy degradation into: (i) clients-with-stale-A entering an A-round, (ii) clients-with-stale-B entering a B-round. Hypothesis (informed by RoLoRA's theory): stale-A is structurally more harmful because A determines the subspace.
 - Plot the analog of paper's Figure 12 but at 50 clients and across p.

2. **Proposed fix (week 7):** Two simple variants, choose based on characterization:
 - **A-skip:** if too few clients have fresh A in an A-round (below threshold τ), do a B-round instead.
 - **B-only catch-up:** stragglers always upload B regardless of round parity; A-rounds aggregate only over fresh-A clients.
 - **Asymmetric-rate alternation:** if A is more sensitive, do 1 A-round per 2 B-rounds at low participation.

3. **Analysis ties to paper's Theorem (week 8):** RoLoRA's exponential-convergence proof for the linear regression case assumes synchronous full participation. Re-derive (or numerically illustrate on the toy MNIST setup) what happens with random participation. Even a back-of-envelope theorem with a participation-aware error term is workshop-quality.

4. **Secondary experiments (week 6–7) for the report:**
 - LoRA+ asymmetric LR sweep {1, 4, 8, 16, 32} × RoLoRA at rank 4, 50 clients, MNLI.
 - Warmup-then-freeze {5%, 10%, 20%, 50%} on 3-client and 50-client MNLI.
 - Optimizer-state-at-block-switch ablation (reset Adam vs persist) — directly addresses FedMomentum's critique.

This package gives you: a clean reproduction (Section 4), a characterization with novel diagnostic plots (Section 5), a method (Section 6), and two clean ablations (Section 7). It is *exactly* the shape of an FL@FM-NeurIPS or ES-FoMo workshop paper.

---

## Caveats

- I could not directly verify the contents of the OpenReview supplementary (the OpenReview forum page returned 403 to fetching). Plan A is to download from the official OpenReview portal once the team is logged in. Plan B (fork FedSA-LoRA) is robust to whatever you find.
- Compute estimates are inferred from typical RoBERTa-Large LoRA throughput (Microsoft's original LoRA: 4×V100 for GLUE in hours; FedEx-LoRA reports rank-4 RoBERTa-Large rounds ~minutes on A100; LoRA-XS used 2×A100/H100 for similar workloads). Single-GPU simulator overhead is the unknown — if the paper's code instantiates 50 client model copies in memory, you'll need 1×A100 80 GB even for RoBERTa; if it runs sequentially with one model copy, 10 GB is enough. Best case 1.5×, worst case 3× my estimates. Treat all hour numbers as ±50%.
- The improvement angle ranking reflects the literature as of arXiv preprints through early March 2026 (last papers I confirmed: FedRot-LoRA Feb 27 2026, FedMomentum Mar 10 2026, SFed-LoRA Mar 2026). New entries between March and your week 1 (May 2026) could narrow the gap further; **do a fresh arXiv pass on Day 1** with queries like "federated LoRA alternating partial participation" and "RoLoRA" to confirm angle (a)+(g) is still uncrowded.
- Llama-2-7B reproductions of Table 2/3 in the paper are weakly reproducible at the absolute-number level across many papers; treat that as stretch and report relative deltas to your own LoRA baseline.
- The paper has multiple arXiv versions (v1 Feb 2025, v2 Feb 2025, v3 Sep 2025, v4 Oct 2025). The NeurIPS camera-ready (v4) is the version to reproduce. Confirm the table numbers you target are from v4 specifically, not v1.
- Do not chase 100% match on every cell. The course graders care about (i) defensible reproduction methodology, (ii) clear analysis, (iii) original contribution. A reproduction within ±2% with documented seed variance is stronger than a chased 0.1% match without confidence intervals.