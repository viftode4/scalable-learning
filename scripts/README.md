# Scripts

One-off prep utilities. Run from the repo root.

| Script | What it does |
|---|---|
| `prep_glue.py` | Download MNLI/QQP/QNLI from HuggingFace and tokenize once with `roberta-large` to `$SLS_DATA_DIR`. Idempotent. Use `--dry-run` for a no-op check. |
| `extract_supplement.sh` | Unzip the OpenReview RoLoRA supplement (`~/Downloads/rolora-supplement.zip` by default) into `code/harness/rolora-supplement/` and print a SHA256 + content sanity check. |

Run via uv to use the pinned env:

```bash
uv run python scripts/prep_glue.py --task mnli --dry-run
bash scripts/extract_supplement.sh
```
