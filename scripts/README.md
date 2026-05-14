# Scripts

One-off prep utilities. Run from the repo root.

| Script | What it does |
|---|---|
| `prep_glue.py` | Download MNLI/QQP/QNLI from HuggingFace and tokenize once with `roberta-large` to `$SLS_DATA_DIR`. Idempotent. Use `--dry-run` for a no-op check. |
| `extract_supplement.sh` | Unzip the OpenReview RoLoRA supplement (default looks at `~/Downloads/rolora-supplement.zip`, but the actual file is named `5662_Robust_Federated_Finetuni_Supplementary Material.zip` — pass the path explicitly) into `code/harness/rolora-supplement/` and print a SHA256 + content sanity check. |
| `install_supplement.sh` | Create the supplement's isolated Python 3.9 venv at `code/harness/rolora-supplement/RoLoRA-code/.venv-supplement`, apply `code/harness/rolora-supplement.patch` (adds `SLS_ALTERNATION_MODE=rolora\|lora\|ffa_lora` env-var switch), pip-install `federatedscope` + pinned LLM extras, and import-test the result. Idempotent. |

Run via uv to use the pinned env:

```bash
uv run python scripts/prep_glue.py --task mnli --dry-run
bash scripts/extract_supplement.sh
```
