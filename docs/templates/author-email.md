# Author email — contingency template

Send this only if the OpenReview supplement (`u4mobiHTJl`) does not let us reproduce the LoRA baseline on 3-client MNLI within ±2% of paper Table 1. The deep-research plan recommends Khisti's address as the contact most likely to respond.

---

**To:** akhisti@ece.utoronto.ca
**Cc:** shuangyi.chen@mail.utoronto.ca, yuanxin.guo@mail.utoronto.ca
**Subject:** Question about reproducing RoLoRA (NeurIPS 2025) — TU Delft course project

Dear Prof. Khisti,

We are a three-student team in TU Delft's CS 4725 "Scalable Learning Systems" research seminar, working to reproduce and extend your RoLoRA paper (NeurIPS 2025) as our course project. We have been working from the supplementary materials attached to the OpenReview submission `u4mobiHTJl`.

We are reaching out because we have run into a reproducibility issue we cannot resolve from the supplement alone. Specifically:

- Setting: RoBERTa-Large, MNLI, **3 clients**, rank 4, batch size 32, 20 local epochs, 30 communication rounds, seed 0.
- Method: plain LoRA (FedAvg of A and B separately) as a baseline.
- Paper value (Table 1, top row): **<paper number>**
- Our reproduction: **<our number>** (off by **<delta>**% with the matching hyperparameters as best we can read from the supplement).

We have verified that:
- the aggregation is `avg(A)` and `avg(B)` separately (the expected baseline bug),
- the initial weights match (B = 0, A Kaiming),
- the eval is on the dev split,
- the Dirichlet split is IID at 3 clients.

A couple of specific questions if you have a moment:

1. Are there hyperparameters (LR schedule, dropout, warmup, weight decay) not stated in the supplement that we should match?
2. Is the "20 local epochs" in the paper an epoch count or a SGD-step count?
3. Would it be possible to release a polished GitHub repo of the RoLoRA code at any point? It would help our extension experiments (we plan to add (a) orthogonal/SVD initialisation for A, (b) LoRA+-style separate learning rates for A and B, and (c) an adaptive server-side optimiser, all preserving the alternating structure).

We fully understand if a more polished release is not currently planned — any pointers on the items above would already be very useful.

Thank you for considering this. Happy to share our reproduction code or run any specific configuration you'd like us to verify.

Best regards,

Vlad Iftode, Daniel Popovici, Sorin Zele
TU Delft CS 4725

---

## Notes for the sender
- Fill the **bold placeholders** before sending — exact numbers, not approximations.
- Attach our reproduction config YAML or paste the relevant snippet.
- Set a one-week internal deadline (per deep-research plan); if no response by then, proceed with the FedSA-LoRA fallback and note the pivot in `docs/decisions/`.
