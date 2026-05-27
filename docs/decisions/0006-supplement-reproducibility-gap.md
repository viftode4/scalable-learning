# ADR 0006 — Reproducibility gap between the OpenReview supplement and paper Table 1

**Status:** Accepted (2026-05-27). All 6 matrix arms + control complete;
empirical-evidence table below is final.

## Context

ADR 0001/0004 commit us to the OpenReview supplement
(`5662_Robust_Federated_Finetuni_Supplementary Material.zip`, SHA256
`ca9a64cb64bb48bb0a6dc35179760fe0e561dd566a474877ab42f65453aa4c11`) as
the primary reproduction artifact for Chen et al., *Robust Federated
Finetuning of LLMs via Alternating Optimization of LoRA* (NeurIPS 2025).
There is **no public github repository** for this paper — OpenReview is
the only released artifact (verified 2026-05-27: the github URL
`huangowen/rolora` is a different paper on weight-activation
quantization, not the federated alternating-LoRA work).

Across our local QNLI runs (`results/overnight_*.log`) and Daniel's
cluster runs (`evidence/cluster_runs/{9971857,9976252}/`) the supplement
as published does not reach the paper's reported Table 1 accuracies on
QNLI. The cluster jobs sat at chance accuracy after running 30 rounds at
the authors' shipped hyperparameters; the laptop runs reproduced the
same flat-loss / chance-test-accuracy behaviour at smaller scale. This
ADR records what we found, what is and isn't a real gap, and what we
have to do in our report to defend that distinction.

## What `test_glue.yaml` ships

`federatedscope/llm/baseline/test_glue.yaml` is the authors' only
QNLI/RoBERTa config in the supplement. Key settings:

- `model.type: roberta-large@huggingface_llm` — loaded via
  `AutoModelForSequenceClassification(num_labels=2)`
  (`model_builder.py:21`). The roberta-large checkpoint contains no
  SEQ_CLS head, so Hugging Face creates a **fresh, randomly initialised
  classifier head** on every load (the "Some weights of
  RobertaForSequenceClassification were not initialized from the model
  checkpoint…" warning).
- `federate.client_num: 50`, `federate.total_round_num: 30`,
  `federate.share_local_model: True`.
- LoRA `r: 8`, `lora_alpha: 32`, `lora_dropout: 0.1`.
- `train.local_update_steps: 20` (batch-mode), `dataloader.batch_size: 32`.
- `train.optimizer.lr: 0.005`, `weight_decay: 0.0002`,
  `type: Adam` **commented out** (line 74). The default optimizer
  builder therefore returns `torch.optim.SGD`, so the shipped recipe is
  **SGD with `lr=0.005`** — *not* the Adam/AdamW that LoRA fine-tuning
  papers usually rely on.

## What the supplement's trainer does

`federatedscope/llm/trainer/trainer.py:_hook_on_fit_start_init` (lines
135–215 in the version we extracted, line numbers shifted slightly after
our fix):

1. Build a fresh optimizer from `ctx.model` (no `requires_grad` filter,
   so every parameter is in the optimizer's `param_groups`).
2. Apply the per-mode factor freezing
   (`alternation_mode in {rolora, lora, ffa_lora}`).
3. Run the block `if self.step_count == 0: Freeze classifier` —
   permanently freezes BOTH copies of PEFT's
   `ModulesToSaveWrapper` (`original_module.*` and
   `modules_to_save.default.*`). There is **no documentation anywhere in
   the supplement** explaining this block; no README, no config flag,
   no comment.
4. Increment `self.step_count`.

Quirks we discovered while reproducing:

- The hook fires for every `cur_mode` (train, val, test), so
  `self.step_count` actually advances 3× per round per client (verified
  by the original log: 36 alternation prints across 4 rounds at 3
  clients). The `step_count % 2` rule still produces the right
  train-time alternation by luck of arithmetic.
- `share_local_model: True` means all clients reference the same model
  object; sequential clients within a round see the shared model
  mutating in place as earlier clients finish training. This is a
  meaningful deviation from textbook FedAvg semantics (independent
  local copies, then average), even though the *active-factor*
  averaging invariant is preserved (only one factor moves per round, so
  the frozen factor is identical across clients regardless of order).

## What we changed

Two patches on the `fix-rolora` branch (commits `8c60faa`, `3e5f68e`):

| Patch | File | Effect |
|---|---|---|
| Classifier-unfreeze (Daniel) | `trainer.py` | Remove the
`if step_count==0: Freeze classifier` block; explicitly set
`'classifier' in name → requires_grad = True` every TRAIN round.
Default behaviour. To reproduce the unmodified supplement, set
`SLS_FREEZE_CLASSIFIER=1`. |
| Scope alternation to TRAIN mode + build optimizer after alternation +
single-client mech probe + opt-in debug prints (Vlad) | `trainer.py`,
`client.py` | `step_count` and `requires_grad` flips only happen in
train fits (was happening in val/test too, drifting `step_count` 3×).
Optimizer built after the alternation block. wandb mech probe gated to
`self.ID == 1` so multi-client logs at the same wandb step don't
overwrite the start-of-round value with mid-round shared-model
mutations. |

A per-batch grad probe (`SLS_DEBUG_GRAD=1`) confirms the alternation is
exact on the patched trainer: in B-train rounds `A.grad is None` for
all 24 LoRA-A params, in A-train rounds `B.grad is None` for all 24
LoRA-B params.

## Empirical evidence (RoBERTa-base QNLI, 3 IID clients, 40 rounds, mech-check scale)

| Arm | Trainer | Optimizer | Classifier | r1 test_acc | r39 test_acc | Log |
|---|---|---|---|---|---|---|
| `rolora_sgd` | patched | SGD lr 0.005 (shipped) | unfrozen | 0.5054 | **0.5162** | `results/overnight_rolora_sgd.log` |
| `lora_sgd` | patched | SGD lr 0.005 | unfrozen | 0.5054 | **0.5213** | `results/overnight_lora_sgd.log` |
| `ffa_lora_sgd` | patched | SGD lr 0.005 | unfrozen | 0.5054 | **0.5193** | `results/overnight_ffa_lora_sgd.log` |
| `rolora_adamw` | patched | AdamW lr 5e-4 | unfrozen | 0.5054 | **0.8766** | `results/overnight_adamw_40.log` |
| `lora_adamw` | patched | AdamW lr 5e-4 | unfrozen | 0.5054 | **0.8783** | `results/overnight_lora_adamw.log` |
| `ffa_lora_adamw` | patched | AdamW lr 5e-4 | unfrozen | 0.5054 | **0.8607** | `results/overnight_ffa_lora_adamw.log` |
| `control_originalfreeze_40` | upstream freeze + scope fix | AdamW lr 5e-4 | **frozen (shipped)** | 0.5960 | **0.8688** | `results/overnight_control_originalfreeze_40.log` |

Headline: AdamW lr 5e-4 reaches **0.86-0.88** across all three modes
(rolora 0.8766, lora 0.8783, ffa_lora 0.8607); SGD lr 0.005 stays at
**chance (0.49-0.52)** across all three modes (rolora 0.5162, lora
0.5213, ffa_lora 0.5193). The classifier-freeze block costs <0.01
absolute test_acc (rolora_adamw 0.8766 vs control 0.8688), confirming
it's a minor code-quality issue, not the killer.

**Caveat that must appear in the report:** at this small mech-check
scale (3-client IID, RoBERTa-base, 40 rounds) the three AdamW arms
are within 0.02 of each other — `lora_adamw` (0.8783) even slightly
edges `rolora_adamw` (0.8766). This is *expected*: RoLoRA's
"exact alternating aggregation" advantage manifests when client count
is high enough for FedAvg-of-AB to introduce meaningful interference,
not at 3 IID clients on a small backbone. So this matrix neither
confirms nor refutes the paper's mode ranking — it only confirms the
SGD-vs-AdamW story. The cluster cells (20- and 50-client
RoBERTa-Large) are the actual test of the paper's claim.

(Cluster: Daniel's `9971857` and `9976252` jobs used roberta-large +
SGD lr 0.005 + classifier-unfreeze patch and hit the 4 h wall-time
ceiling with test_acc stuck at chance.)

## What the paper actually says

Reading `docs/research/paper-rolora.pdf` (full audit pages 1-50)
sharpens the gap from "the code is broken" to "the paper underspecifies
the recipe AND the shipped artifact picks unfortunate defaults":

- **Table 6 (page 41) — the paper's "Hyper-parameters for GLUE task"
  table — only lists `Total comm. rounds`, `Batch Size`, `Local
  Epochs`.** It does NOT list optimizer, learning rate, weight decay,
  or scheduler. The reader has no way to know which optimizer to use
  from Table 6 alone.

- **Page 7 (Section 5, "Implementation & Configurations"):**
  > "Specifically, the learning rate is chosen from the set
  > {5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1}."

  The paper sweeps LR over 8 values per dataset and reports
  *best-on-test-averaged-over-seeds*. The shipped `test_glue.yaml` uses
  **lr 0.005**, which is one of those 8 values — not necessarily the
  best for QNLI. The optimizer is still not stated, but the upper end
  of the LR range (5e-2, 1e-1) only makes sense for SGD; AdamW would
  diverge at 1e-1. So we **infer** SGD was their default.

- **Table 6's round counts are for 3 clients.** The note below the
  table reads: "When increasing the number of clients, we decrease the
  total communication rounds accordingly to maintain a constant sample
  count used during fine-tuning." So QNLI is 500 rounds at 3 clients,
  scaling to ~75 rounds at 20 clients and ~30 rounds at 50 clients.
  The shipped `test_glue.yaml` (50 clients × 30 rounds) matches the
  50-client setting. **Daniel's cluster `repro_qnli_c20_r4_*.sbatch`
  ran 30 rounds at 20 clients, which is under the paper's implied 75
  rounds.** Two compounding under-tuning problems on the cluster: weak
  optimizer + too few rounds.

- **NeurIPS checklist (page 49, Q5 "Open access to data and code")**
  answers "Yes — datasets are open-source; the code is uploaded". The
  uploaded code is the supplement we have. So the checklist obligation
  is technically discharged, but the shipped config doesn't pin enough
  hyperparameters to recover the reported numbers without an LR sweep.

- **Paper's Table 1 cell for QNLI / RoBERTa-Large / 50 clients / rank
  4: RoLoRA 90.00 ±0.61.** That's the number our cluster cells should
  aim for. We reach 0.86-0.88 on a smaller (roberta-base, 3-client,
  40-round) setup with AdamW lr 5e-4; scaling to roberta-large +
  appropriate rounds should close most of the gap.

## What this means for the cluster (concrete recommendations)

1. **Re-launch `slurm/repro_qnli_c20_r4_*.sbatch` with
   `train.optimizer.type AdamW train.optimizer.lr 0.0005` overrides AND
   `federate.total_round_num 75`** to match the paper's implied
   20-client setting. Without both fixes the cluster job will not
   produce paper-comparable evidence.

2. **For the C3 50-client cell, keep 30 rounds** (matches the
   shipped/paper recipe) but still swap optimizer to AdamW lr 5e-4.

3. **For the C1 3-client cell, use 500 rounds** at AdamW lr 5e-4 — but
   note this is the most expensive cell, so submit it last.

4. **If time permits, sweep LR over the paper's set
   {5e-4, 1e-3, 2e-3, 5e-3} with AdamW** to find the best for our
   exact setup before claiming a paper-comparable number. The paper
   reports best-of-sweep; honest reproduction does the same.

## What this means for the report

We have **direct empirical evidence** for one strong reproducibility
gap and one weaker one. We must keep them separate in writing.

### Gap 1 — Shipped optimiser is far too weak (strong, defensible)

The shipped `test_glue.yaml` uses SGD lr 0.005. With this optimiser the
model cannot learn QNLI in 30 rounds at either roberta-base (laptop,
mech-check) or roberta-large (cluster, real reproduction scale).
Switching to AdamW lr 5e-4 unblocks learning to ≥ 0.87 test_acc on
roberta-base in 40 rounds. The paper text needs to be checked for
optimiser claims; a 30-round SGD-lr-0.005 recipe is unusual for
LoRA-style PEFT fine-tuning, where AdamW lr 1e-4–5e-4 is standard.

### Gap 2 — Undocumented classifier-freeze block (weaker, code-quality concern)

`trainer.py` permanently freezes the SEQ_CLS head from round 0 onward
with no documentation. Setting `SLS_FREEZE_CLASSIFIER=1` reproduces this
behaviour; the control run currently shows the model can still partially
learn (0.82 at round 9 with AdamW), presumably because LoRA adapts
the upstream features into the random classifier's effective decision
direction. So the freeze costs accuracy and convergence speed but does
not by itself pin QNLI at chance. We should frame this as a code-quality
finding ("the published supplement contains an undocumented setting
that is surprising and unmotivated; reproducing the paper requires
disabling it") rather than a "the code can't learn" claim.

### What to NOT claim

- Do **not** claim "the published code cannot learn beyond chance" —
  the control experiment refutes that.
- Do **not** claim "the alternation logic is broken" — once the
  eval-mode `step_count` drift is fixed, the alternation is provably
  exact (per the `SLS_DEBUG_GRAD=1` probe).
- Do **not** claim the github repository is the canonical source — the
  federated RoLoRA paper has no public github; the OpenReview
  supplement is the only released artifact.

## Decisions

1. **Make this gap a first-class section of the final report.** Add a
   "Reproducibility audit of the OpenReview supplement" section between
   §3 (Reproduction protocol) and §4 (Local sanity evidence). The
   section must (a) cite `test_glue.yaml` line numbers and (b) include
   the empirical-evidence table once the overnight matrix finishes.
2. **Use the fixed trainer for every paper-scale experiment we run on
   the cluster from 2026-05-27 onward.** Existing `slurm/repro_*_sgd*`
   sbatch files need to be re-launched after switching to AdamW
   lr 5e-4 (or after we confirm a working SGD recipe — but the burden
   of proof is now on SGD).
3. **Keep `SLS_FREEZE_CLASSIFIER=1` as an opt-in env-var gate.** Useful
   for the control experiment and the report's reproducibility section.
4. **Update `README.md` and `docs/progress.md`** to surface the finding
   prominently so neither future agents nor teammates re-launch
   experiments at the shipped hyperparameters.

## Consequences

- The C2 / C3 / C4 cluster cells in `docs/experiment-matrix.md` must
  re-run with AdamW lr 5e-4 (and possibly retuned wd / schedule).
  The previous cluster evidence in `evidence/cluster_runs/` is
  preserved as part of the audit, not as primary reproduction
  evidence.
- The report's "Reproduction protocol" section now has to explicitly
  list the two patches we apply to the supplement and the optimiser
  swap. This was always part of the discipline but now becomes a
  contribution rather than a footnote.
- If we cannot reproduce the paper's accuracies even after the optimiser
  fix, the report frames that as a strong second-order claim: "with
  the supplement plus the minimal set of fixes required to make it
  learn at all, we obtain test accuracies of X on cell Y, Z points
  below the paper's reported number". That's still a substantive
  result.

## Open follow-ups

- After the overnight matrix lands, complete the empirical-evidence
  table above and promote this ADR to Accepted.
- Read the paper text (`docs/research/paper-rolora.pdf`) for the exact
  optimiser claim. If the paper claims Adam/AdamW lr 1e-4–5e-4 and the
  supplement ships SGD lr 0.005, the gap is unambiguous. If the paper
  claims SGD lr 0.005, we need to investigate why the same recipe gives
  chance on our hardware.
- Investigate whether the freeze block has ANY useful function (e.g.,
  was it added to prevent server-side aggregation from corrupting an
  already-tuned head when continuing from a checkpoint?) — currently
  we have no hypothesis for why it exists.
- Decide whether to disable `share_local_model: True` for the
  improvement experiments. The shared-model setup is consistent with
  the supplement's published code, but textbook FedAvg uses
  independent local copies; this is a separate analysis axis.
