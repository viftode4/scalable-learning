# Harness checkouts

Two harnesses, used in priority order. See `docs/decisions/0001-harness-strategy.md` for the rationale.

## `rolora-supplement/` — primary (authors' code)
The OpenReview supplementary zip for RoLoRA (forum `u4mobiHTJl`). Required by the project proposal ("we will use the authors' released code as the starting point").

**Audit summary** (see `docs/decisions/0004-supplement-audit.md` for the full audit):
- Zip SHA256 (verify after download): `ca9a64cb64bb48bb0a6dc35179760fe0e561dd566a474877ab42f65453aa4c11`.
- License: **Apache 2.0** (redistributable; vendored here so teammates get the same source).
- Built on **FederatedScope-LLM** (~800 Python files, 16 MB).
- `federatedscope/llm/trainer/trainer.py` contains the alternation logic, gated on a hardcoded `self.save_mode = True` flag with `step_count`-parity for round selection.
- **No LoRA / FFA-LoRA baseline modes** in the supplement — the alternation is the only path. The tracked `rolora-supplement.patch` adds `rolora` / `lora` / `ffa_lora` modes so all three branches can run in the same harness.
- Example config: `federatedscope/llm/baseline/test_glue.yaml` (RoBERTa-Large, QNLI, 50 clients, rank 8, 30 rounds).
- Suggests Python 3.9 + PyTorch 2.0; our main env is 3.11 + 2.3. The isolated `.venv-supplement` env is created by `make install-supplement`.

**Vendored in git.** Fresh clones already contain the authors' code. Build the isolated runtime with:
```bash
bash scripts/install_supplement.sh
```

To refresh from the original OpenReview zip, use `bash scripts/extract_supplement.sh /path/to/zip` and re-run the audit.

Run local smoke checks with our mode switch:
```bash
make supplement-smoke MODE=rolora
make supplement-smoke-all
```

Swap `SLS_ALTERNATION_MODE` / `MODE` between `rolora`, `lora`, `ffa_lora` to compare the three methods on the same harness. The patch lives at `code/harness/rolora-supplement.patch` (tracked in git, idempotently re-applied by `install_supplement.sh`). Use `scripts/run_supplement.py` instead of calling `federatedscope/main.py` directly on macOS; it preserves the upstream behavior but avoids the hostname-resolution crash in FederatedScope logging.

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
