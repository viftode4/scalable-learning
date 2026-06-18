<!-- Paste-ready LaTeX for the RoLoRA report revision. Companion to docs/report-review-panel.md.
     Every number traces to runs/REGISTRY.md / docs/decisions/0006-supplement-reproducibility-gap.md /
     evidence/toy_heterogeneity_20260603/. The two `% VERIFY` lines are the only numbers NOT in our
     own ground-truth (they come from the original paper's Table 1) — confirm before submitting. 2026-06-18. -->

# RoLoRA report — paste-ready LaTeX revision blocks

Each block is placed at the indicated location in the Overleaf source. Grounding
for every number is given in the provenance table below.

## Number provenance (single source of truth)

| Arm (QNLI proxy, RoBERTa-base, 50 clients, rank 4, lr 1e-2, final round) | Value | Source |
|---|---|---|
| RoLoRA (n=3: 82.14/87.13/86.04) | 85.10 ± 2.97 (CI95) | REGISTRY §1 / §2 |
| Orthogonal A alone (n=1) | 82.94 (−2.16) | REGISTRY §1 line 25, 35 |
| SVD-compensated (n=3: 85.15/86.53/85.43) | 85.70 ± 0.82 (CI95) | REGISTRY §1 line 23 |
| Adaptive refresh + orth A (n=3: 86.56/86.36/87.09) | 86.67 ± 0.43 (+1.57) | REGISTRY §1 line 21 |
| Orthogonal A + BBA (n=3: 88.54/88.10/86.16) | 87.60 ± 1.43 (+2.50) | REGISTRY §1 line 20 |
| Welch, orth+BBA vs RoLoRA | t≈1.5, p≈0.24 (n=3) | recomputed 2026-06-18 |
| SVD LR sweep | 5e-3→83.76±1.40 / 1e-2→85.70±0.82 / 2e-2→61.44±19.61 (2/3 at chance) | REGISTRY §2b |
| Baselines across sweep {5e-3,1e-2,2e-2} | ~48–55% final, never >~55% | REGISTRY §2 |
| Audit: SGD lr0.005 (shipped) | RoLoRA 0.5162 / LoRA 0.5213 / FFA 0.5193 | ADR-0006 |
| Audit: AdamW lr5e-4 | RoLoRA 0.8766 / LoRA 0.8783 / FFA 0.8607 | ADR-0006 |
| Audit: frozen vs unfrozen classifier | 0.8688 vs 0.8766 (<0.01 cost) | ADR-0006 |
| Toy orth-A (n=5, heterogeneous) | base_rolora 0.8389 (sd 2.04pp) → orth 0.8719 (sd 1.31pp) | evidence/toy_heterogeneity_20260603 |
| Toy significance (recomputed from raw seeds) | Welch t=3.05 p≈0.02; paired t=2.57 p≈0.06; var F=2.44 | computed 2026-06-18 |

**`% VERIFY` flags** — the original-paper numbers (RoLoRA-large QNLI 90.00; LoRA
QNLI ≈78; MNLI ≈chance) are from our review's read of the original Table 1 and are
NOT in our registry. Confirm against the original paper before paste.

---

## Block 1 — §3.2 "default configuration" sentence

**Where:** §3.2 "Model and code", replace the final sentence.

```latex
We build on the code released by the RoLoRA authors rather than reimplementing the
training pipeline. The released configuration does not by itself reproduce the
reported accuracies; the corrections we applied are described in
Section~\ref{sec:repro-audit}. All QNLI results in this report use those
corrections rather than the shipped defaults.
```

---

## Block 2 — Reproducibility audit (short main body + appendix)

**Where (main body):** new subsection at the end of §3. Add `\label{sec:repro-audit}`.

```latex
\subsection{Reproducing the Released Artifact}
\label{sec:repro-audit}
The federated RoLoRA paper provides no public code repository; the OpenReview
supplement is its only released artifact. The supplement runs, but it does not
pin the optimiser, and its single learning rate does not reproduce the reported
QNLI results: the hyperparameter table lists only the number of rounds, batch
size, and local epochs, and the released configuration defaults to SGD with
$\eta=0.005$. Under this setting all three methods remain at chance in a mechanism
check (RoBERTa-base, three IID clients). We therefore recovered a working recipe.
As no optimiser or per-configuration learning rate is given, we replaced the
optimiser with AdamW, which restores test accuracy to $0.86$--$0.88$, and selected
the learning rate using the sweep in Section~\ref{sec:lm-repro}. We made two
further changes to the trainer. First, we restrict the factor alternation to
training steps; the released code also runs it during evaluation, which advances
the step counter and flips the trainable factor outside training. Second, we
remove an undocumented freeze of the classifier head; this is a code-quality
change, and we measure its effect at under $0.01$ accuracy. All QNLI results
reported here use this recipe; configuration details and supporting measurements
are given in Appendix~\ref{app:repro}.
```

**Where (appendix):** after `\appendix`. Add `\label{app:repro}`. Give §4.1.2 the
label `\label{sec:lm-repro}` so the cross-references resolve.

```latex
\section{Reproducibility Audit Details}
\label{app:repro}
The supplement's only QNLI configuration,
\texttt{federatedscope/llm/baseline/test\_glue.yaml}, sets
\texttt{train.optimizer.lr: 0.005} with the Adam option commented out (line~74),
so the default builder selects SGD; the optimiser is not stated in the paper. The
trainer also freezes the classifier head from the first step, without
documentation. Our two changes restrict the factor alternation to training steps,
since the released code also alternates during evaluation and advances the step
counter threefold per round, and unfreeze the classifier. A per-batch gradient
check confirms that the alternation remains exact after these changes: in
$B$-rounds the gradient of $A$ is null, and in $A$-rounds the gradient of $B$ is
null. The model learns once the optimiser is corrected, so we do not attribute the
chance-level behaviour to either change.

\begin{table}[h]
\centering
\caption{Optimiser check (RoBERTa-base, three IID clients, 40 rounds, final test
accuracy). SGD is the released recipe; AdamW is our correction.}
\begin{tabular}{lccc}
\toprule
Optimiser & RoLoRA & FedAvg-LoRA & FFA-LoRA \\
\midrule
SGD $\eta{=}0.005$ (shipped) & 0.516 & 0.521 & 0.519 \\
AdamW $\eta{=}5\!\cdot\!10^{-4}$ & 0.877 & 0.878 & 0.861 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Learning-rate sweep on the QNLI proxy (RoBERTa-base, 50 clients, rank 4,
final-round accuracy). The baselines remain at chance at every rate; SVD peaks at
$10^{-2}$ and is unstable above it.}
\begin{tabular}{lccc}
\toprule
LR & RoLoRA & SVD-comp. & note \\
\midrule
$5\!\cdot\!10^{-3}$ & 81.47 & $83.76 \pm 1.40$ & stable \\
$10^{-2}$           & 85.10 & $85.70 \pm 0.82$ & optimum \\
$2\!\cdot\!10^{-2}$ & 81.20 & $61.44 \pm 19.61$ & SVD unstable (2/3 at chance) \\
\bottomrule
\end{tabular}
\end{table}

With the classifier freeze enabled (AdamW), RoLoRA reaches $0.869$ against $0.877$
unfrozen, confirming that the freeze costs under $0.01$ accuracy.
```

---

## Block 3 — Baseline-collapse explanation

**Where:** §4.1.2, immediately after the "Outcome" paragraph.

```latex
\textbf{Interpretation of the baseline collapse.} The baselines do not only score
below RoLoRA; they remain near chance ($\approx 52\%$ final, and below $55\%$ at
every learning rate we tried), while RoLoRA reaches $85.1\%$. Two observations
indicate that this is not a tuning artefact. First, the baselines were swept over
the same learning rates $\{5\!\cdot\!10^{-3}, 10^{-2}, 2\!\cdot\!10^{-2}\}$ as
RoLoRA without exceeding chance. Second, the effect is the one RoLoRA is designed
to remove: at $50$ clients with one label per client, the cross-term interference
in~(2) dominates the aggregated update, and within the $20$-round budget
FedAvg-LoRA does not accumulate a consistent signal, while FFA-LoRA is further
limited by its frozen, randomly oriented down-projection. The original paper
reports related behaviour at $50$ clients, where its LoRA baseline is already
unstable on QNLI and near chance on the harder MNLI task. % VERIFY original Table-1
Our reduced setting---RoBERTa-base rather than -large, final-round rather than
best-of-sweep accuracy, and $20$ rounds---lowers the QNLI baselines further. We
therefore interpret the collapse as an amplification of a known fragility under a
reduced budget rather than a contradiction of the original results.
```

---

## Block 4 — Table 1 as a bar chart (mean ± 95% CI)

**Where:** replace the current Table 1 with this figure. Upload
`report/figures/lm_proxy_improvements.pdf` to Overleaf (regenerate with
`scripts/plot_lm_proxy_improvements.py`). The orthogonal-$A$-alone negative cell
(82.94, $n{=}1$) is kept in the companion prose below rather than in the chart;
say the word to add it as a fifth bar.

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.92\linewidth]{figures/lm_proxy_improvements.pdf}
\caption{Language-model proxy improvements on QNLI with RoBERTa-base, 50 clients,
rank 4, learning rate $10^{-2}$. Bars are final-round server test accuracy; error
bars are 95\% CI ($1.96\,s/\sqrt{n}$, $n{=}3$ per arm). The dashed line marks the
plain-RoLoRA baseline. The orthogonal-$A$+BBA arm has the largest mean gain
($+2.5$ points) but its 95\% CI overlaps the baseline, and the difference is not
statistically significant (Welch $t\approx1.5$, $p\approx0.24$).}
\label{fig:lm-proxy}
\end{figure}
```

**Companion sentence** in §4.2.2 (so the orth-$A$-alone cell is discussed, not dropped):

```latex
Orthogonal-$A$ initialisation alone does not improve accuracy at this scale: it
reaches $82.94\%$ (single seed), $2.16$ points below plain RoLoRA. The improvement
appears only when the conditioned basis is combined with a $B$-prioritised
schedule, which is consistent with an interaction between initialisation and
schedule rather than an additive effect of initialisation.
```

---

## Block 5 — §5 Discussion (rewritten; removes the `[TODO]`)

**Where:** replace the entire body of §5.

```latex
\section{Discussion}

\textbf{Reproduction.} RoLoRA reproduces at both scales: under high client counts
and heterogeneous data it remains close to the centralised ceiling, while
FedAvg-LoRA and FFA-LoRA fall well short. The match is qualitative rather than
quantitative. Our RoBERTa-base/QNLI proxy reaches $85.1\%$ against the original's
RoBERTa-large $90.0\%$, % VERIFY original 90.00 cell
and the baselines fall to chance rather than degrading gradually; as discussed in
Section~\ref{sec:lm-repro}, this amplifies a fragility present in the original at
$50$ clients rather than contradicting it, and the shared learning-rate sweep
rules out under-tuning. That the ordering survives a substantial reduction in
model size and training budget indicates that we are observing RoLoRA's robustness
rather than an artefact of a single configuration.

\textbf{Initialisation.} Conditioning the shared basis helps where the default
basis is uninformative. On the toy model there is no pre-trained weight matrix, so
the default basis is poor, and orthogonal-$A$ initialisation raises accuracy from
$84.0\%$ to $87.2\%$ and reduces the run-to-run standard deviation from $2.0$ to
$1.3$ points (Welch $t=3.0$, $p\approx0.02$ over five seeds; $F\approx2.4$ for the
variance reduction; the paired test is borderline, $p\approx0.06$). On RoBERTa the
backbone is pre-trained, and SVD-based initialisation yields a smaller gain
($85.1\%$ to $85.7\%$, within RoLoRA's run-to-run spread of $\pm2.97$). A
learning-rate sweep indicates that this is not a tuning effect: SVD reaches
$83.8\%$ at $5\!\cdot\!10^{-3}$, $85.7\%$ at $10^{-2}$, and becomes unstable at
$2\!\cdot\!10^{-2}$, with two of three seeds at chance. When $W_0$ already encodes
useful structure, re-initialising $A$ from its leading singular directions adds
little, since the directions relevant to QNLI need not coincide with the most
energetic directions of $W_0$.

\textbf{Initialisation and schedule.} The configurations that exceed plain RoLoRA
at language-model scale combine a conditioned $A$ with a schedule that prioritises
the $B$-phase: a fixed BBA schedule reaches $87.6\%$ ($n=3$) and adaptive refresh
reaches $86.7\%$ ($n=3$), both above the $85.1\%$ baseline. Neither component helps
in isolation: orthogonal-$A$ initialisation alone falls below the baseline
($82.94\%$), and RoLoRA's own ablation reports no gain from schedule asymmetry
under the default initialisation. This pattern is consistent with an interaction
between initialisation and schedule, in which the $B$-phase becomes more useful
once $A$ provides a stable basis. We report it as preliminary: across three seeds
the gain ($+2.5$ points) lies within plain RoLoRA's run-to-run spread and is not
statistically significant (Welch $t\approx1.5$, $p\approx0.24$), and we have not
run the default-initialisation-with-BBA control needed to separate a schedule main
effect from the interaction.

\textbf{Interventions that do not help.} Unequal step sizes (LoRA+) do not improve
RoLoRA. The method addresses a step-size imbalance between factors trained
together, whereas RoLoRA updates the two factors in separate rounds, so the ratio
only rescales the $B$-phase relative to the $A$-phase; empirically the symmetric
setting is best ($\lambda=2$ falls just below plain RoLoRA, and larger ratios are
worse). FedProx likewise gives no gain: its proximal term limits the drift of
clients that take many local steps, but a RoLoRA client already updates within a
restricted, half-parameter subspace each round, so the anchor mainly constrains
the single factor being trained and leaves the per-round low-rank restriction
unaddressed ($83.4\%$ versus $84.0\%$).

\textbf{Summary.} RoLoRA composes selectively. It benefits from interventions that
improve the shared low-rank subspace and, at language-model scale and
preliminarily, from unequal update frequency, but not from update-level
modifications such as per-matrix learning rates or proximal regularisation, which
target sources of error that its exact-aggregation guarantee already removes.
```

---

## Still to write (say the word)

- Tightened §6 Conclusion that does not repeat the §5 numbers (cold-read C3).
- §1.6 Contributions paragraph + FedSA-LoRA (Guo et al., ICLR 2025,
  arXiv:2410.01463) citation at the A/B-asymmetry argument.
- Team-contributions rewrite (currently omits the audit and run registry).
