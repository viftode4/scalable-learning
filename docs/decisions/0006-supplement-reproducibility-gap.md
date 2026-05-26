# ADR 0006 — Reproducibility gap between the OpenReview supplement and paper Table 1

**Status:** Proposed (2026-05-27). Promote to Accepted once the 6-arm
overnight matrix and the `SLS_FREEZE_CLASSIFIER=1` control experiment
finish and the empirical-evidence table below is final.

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

| Arm | Trainer | Optimizer | Classifier | round 1 test_acc | round 39 test_acc | Log |
|---|---|---|---|---|---|---|
| `rolora_sgd` | patched | SGD lr 0.005 (shipped) | unfrozen (default) | TBD | TBD | overnight matrix (in flight) |
| `lora_sgd` | patched | SGD lr 0.005 | unfrozen | TBD | TBD | overnight matrix |
| `ffa_lora_sgd` | patched | SGD lr 0.005 | unfrozen | TBD | TBD | overnight matrix |
| `rolora_adamw` | patched | AdamW lr 5e-4 | unfrozen | 0.5054 | **0.8766** | `results/overnight_adamw_40.log` |
| `lora_adamw` | patched | AdamW lr 5e-4 | unfrozen | TBD | TBD | overnight matrix |
| `ffa_lora_adamw` | patched | AdamW lr 5e-4 | unfrozen | TBD | TBD | overnight matrix |
| `control_originalfreeze_40` | original freeze + scope fix | AdamW lr 5e-4 | **frozen (shipped)** | TBD | TBD | `results/overnight_control_originalfreeze_40.log` (round 9 partial: 0.8199) |

(Cluster: Daniel's `9971857` and `9976252` jobs used roberta-large +
SGD lr 0.005 + classifier-unfreeze patch and hit the 4 h wall-time
ceiling with test_acc stuck at chance.)

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
