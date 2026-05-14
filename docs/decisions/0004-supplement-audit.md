# ADR 0004 — Audit of the RoLoRA OpenReview supplement

**Status:** Accepted (2026-05-14)

## Context

Per ADR 0001 the OpenReview supplementary zip is the primary harness. The team fetched it (file: `5662_Robust_Federated_Finetuni_Supplementary Material.zip`, SHA256 `ca9a64cb64bb48bb0a6dc35179760fe0e561dd566a474877ab42f65453aa4c11`) and extracted into `code/harness/rolora-supplement/`. We need to know what we actually have before basing the project on it.

## What's in the supplement

- **Layout:** root is `RoLoRA-code/`. Contains:
  - `federatedscope/` — full FederatedScope framework (FederatedScope-LLM fork, ~800 Python files).
  - `sst2/` — dataset-prep scripts for SST-2, MNLI, QQP, QNLI (`{task}2json.py`).
  - `benchmark/`, `doc/`, `environment/`, `setup.py`, `meta.yaml`.
  - `LICENSE` — **Apache License 2.0** (redistribution OK).
- **Size:** 16 MB, 845 files after extraction. Junk `__MACOSX/` metadata removed.
- **README** (`federatedscope/llm/README.md`) gives the run flow:
  1. `conda create -n fs-llm python=3.9 && conda activate fs-llm`
  2. `conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 …`
  3. `pip install -e .[llm]`
  4. `cd sst2 && python qnli2json.py` (per-task data prep)
  5. `python federatedscope/main.py --cfg federatedscope/llm/baseline/test_glue.yaml`
- **Example config** (`federatedscope/llm/baseline/test_glue.yaml`): 50 clients, 30 rounds, RoBERTa-Large, QNLI, **rank 8**, lora_alpha 32, lora_dropout 0.1, batch 32, 20 local *batches* (not epochs) per round, lr 0.005, weight_decay 2e-4. IID splitter.

## Where the alternation lives

`federatedscope/llm/trainer/trainer.py`, lines 25–90:

```python
def __init__(self, ...):
    ...
    self.step_count = 0
    self.save_mode = True   # ← this is the "RoLoRA on" flag

def _hook_on_fit_start_init(self, ctx):
    ...
    if self.save_mode:
        if (self.step_count % 2) == 0:
            print("Freeze A")
            # iterate parameters → lora_A.requires_grad = False, lora_B.requires_grad = True
        else:
            print("Freeze B")
            # iterate parameters → lora_A.requires_grad = True,  lora_B.requires_grad = False
    if self.step_count == 0:
        # freeze classifier on the first round only
    self.step_count += 1
```

Notes:
- The "RoLoRA on/off" knob is **named `save_mode` in code** (misnomer) and is hard-coded `True` in `__init__`. There is no config-driven way to toggle baselines.
- The round counter is a Python `int` mutated per call to the start-of-fit hook. Per-client across-rounds bookkeeping is implicit through this counter.

## What the supplement does **not** contain

- **No vanilla LoRA baseline path.** `save_mode=True` always alternates. No conditional that runs plain "both factors trainable + FedAvg-of-products."
- **No FFA-LoRA baseline path.** No "freeze A forever" branch.
- **No 5-task GLUE driver script.** Each task is one config edit.
- **No paper-figure plotting code.**
- **No partial-participation / DP / improvement-direction code.**

## Decisions

1. **Keep the supplement gitignored.** Even though Apache-2.0 allows redistribution, vendoring 16 MB into a small course repo bloats clones and confuses provenance. Teammates fetch + extract per `docs/setup/openreview-supplement.md`. The SHA256 above is the shared verification hash.
2. **Modify the supplement's `trainer.py` ourselves to add baseline modes.** Specifically, add a `cfg.train.alternation_mode` knob with values `rolora` (the existing alternation), `lora` (both factors trainable, FedAvg-of-A and FedAvg-of-B separately — the bug baseline), `ffa_lora` (A frozen at init, only train B). All three branches share the same eval / data / aggregator path. This is the cleanest way to keep the comparison honest and is the deep-research plan's recommended approach (~50 LOC patch).
3. **Pin a separate Python env for the supplement.** Its README asks for Python 3.9 + PyTorch 2.0; our main repo env is Python 3.11 + PyTorch 2.3. We will:
   - Keep the main repo env (Python 3.11 + PyTorch 2.3, MNIST sanity check + our own code).
   - Set up a **secondary uv env** specifically for the supplement, in `code/harness/rolora-supplement/.venv-supplement/` (gitignored), pinned to torch 2.0.x for first-attempt compatibility. If it works on torch 2.3 we'll consolidate; if not we keep the split.
4. **Defer code modifications until after the kickoff meeting** so we agree on who owns the trainer patch.

## Consequences

- The "primary harness, FedSA-LoRA fallback" split (ADR 0001) is still correct: we use the supplement for the RoLoRA path and add baseline modes by patching the supplement (preferred) OR use FedSA-LoRA for vanilla LoRA / FFA-LoRA comparisons (acceptable backup).
- We must record the patch as a commit in `code/harness/rolora-supplement/` even though the supplement directory is gitignored. Mechanism: keep a git patch file at `code/harness/rolora-supplement.patch` (in git) that re-applies our trainer modifications onto a fresh extraction. Documented in the harness README.
- Local Python compatibility risk if torch 2.3.1 breaks FederatedScope-LLM's old `transformers`/`peft` pins. Verify in week 4 first thing.

## Open follow-ups

- Confirm whether `requirements-torch1.10*.txt` in `environment/` are stale or whether the conda recipe (PyTorch 2.0) is authoritative. Assume conda recipe.
- Verify `sst2/mnli2json.py` produces a JSON in the format `federatedscope`'s `data.type: 'mnli.json@llm'` expects. (The supplement uses MNLI/QNLI/QQP as JSON via these scripts; HuggingFace `datasets` is not the loader.)
- Locate where FFA-LoRA's "B = 0 at init" convention is enforced (or not) — likely nowhere, since the supplement doesn't have FFA-LoRA mode. We add this when we patch the trainer.
