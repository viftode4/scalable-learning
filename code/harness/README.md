# Harness checkouts

Two harnesses, used in priority order. See `docs/decisions/0001-harness-strategy.md` for the rationale.

## `rolora-supplement/` — primary (authors' code)
The OpenReview supplementary zip for RoLoRA (forum `u4mobiHTJl`). Required by the project proposal ("we will use the authors' released code as the starting point").

**Audit summary** (see `docs/decisions/0004-supplement-audit.md` for the full audit):
- Zip SHA256 (verify after download): `ca9a64cb64bb48bb0a6dc35179760fe0e561dd566a474877ab42f65453aa4c11`.
- License: **Apache 2.0** (redistributable, though we keep it gitignored to avoid bloat).
- Built on **FederatedScope-LLM** (~800 Python files, 16 MB).
- `federatedscope/llm/trainer/trainer.py` contains the alternation logic, gated on a hardcoded `self.save_mode = True` flag with `step_count`-parity for round selection.
- **No LoRA / FFA-LoRA baseline modes** in the supplement — the alternation is the only path. To run the headline LoRA-cliff comparison we must add the two baseline branches ourselves (planned ~50 LOC patch to `trainer.py`).
- Example config: `federatedscope/llm/baseline/test_glue.yaml` (RoBERTa-Large, QNLI, 50 clients, rank 8, 30 rounds).
- Suggests Python 3.9 + PyTorch 2.0; our main env is 3.11 + 2.3. A secondary uv env may be needed if compat breaks.

**Not in git** — see `.gitignore`. To get it on a fresh clone:
```bash
# 1. download the zip per docs/setup/openreview-supplement.md (one-time, user-driven)
bash scripts/extract_supplement.sh "/path/to/5662_Robust_Federated_Finetuni_Supplementary Material.zip"
# 2. build isolated venv + apply sls-rolora trainer patch + import-test
bash scripts/install_supplement.sh
```

Run an experiment with our mode switch:
```bash
SLS_ALTERNATION_MODE=rolora \
  code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/bin/python \
  code/harness/rolora-supplement/RoLoRA-code/federatedscope/main.py \
  --cfg code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/baseline/test_glue.yaml
```

Swap `SLS_ALTERNATION_MODE` between `rolora`, `lora`, `ffa_lora` to compare the three methods on the same harness. The patch lives at `code/harness/rolora-supplement.patch` (tracked in git, idempotently re-applied by `install_supplement.sh`).

## `fedsa-lora/` — backup harness (git submodule)
Git submodule pointing at our fork `viftode4/FedSA-LoRA` (upstream: `Pengxin-Guo/FedSA-LoRA`, ICLR 2025).

Why it's here: FedSA-LoRA already implements LoRA, FFA-LoRA, and FedSA-LoRA on RoBERTa-base/large for GLUE under FedAvg with Dirichlet-α non-IID splits. The deep-research plan flags that the OpenReview supplement may be a research-grade dump; FedSA-LoRA is a vetted public alternative that the W2 kill criterion explicitly authorizes us to pivot to.

```bash
# Initialize after clone
git submodule update --init --recursive

# Pull upstream changes from Pengxin-Guo/FedSA-LoRA into our fork:
cd code/harness/fedsa-lora
git remote add upstream https://github.com/Pengxin-Guo/FedSA-LoRA.git  # one-time
git fetch upstream
git merge upstream/main          # or rebase
git push                          # pushes to viftode4/FedSA-LoRA
cd ../../..
git add code/harness/fedsa-lora && git commit  # records the new submodule SHA
```

When we start modifying the submodule, create a branch in `viftode4/FedSA-LoRA` named `sls-rolora/main` and point the submodule at it.

## Do not clone
- `HuangOwen/RoLoRA` — different paper (EMNLP'24, quantization).
- `alibaba/FederatedScope/tree/llm` — heavy, older codebase; the deep-research plan rejects it as the main vehicle.
