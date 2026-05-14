# Scripts

Small setup/run utilities. Run from the repo root.

| Script | What it does |
|---|---|
| `prep_glue.py` | Download MNLI/QQP/QNLI from HuggingFace and tokenize once with `roberta-large` to `$SLS_DATA_DIR`. Idempotent. Use `--dry-run` for a no-op check. |
| `extract_supplement.sh` | Unzip the OpenReview RoLoRA supplement into `code/harness/rolora-supplement/`. With no args, it checks the real OpenReview filename in `~/Downloads` first, then legacy fallback names. |
| `install_supplement.sh` | Create the supplement's isolated Python 3.9 venv, apply `code/harness/rolora-supplement.patch`, install pinned FederatedScope/LLM deps, and import-test the result. Idempotent. |
| `run_supplement.py` | Launcher for the supplement's `federatedscope/main.py`; keeps a small macOS hostname compatibility patch outside the vendored code. |
| `smoke_supplement.sh` | Run `experiments/configs/smoke_supplement.yaml` for one or more modes: `rolora`, `lora`, `ffa_lora`, or `all`. |
| `local_suite.py` | Orchestrate local evidence suites: `smoke` and `full-local`. Use `--dry-run` to print the exact commands. |

Common commands:

```bash
make check
make mnist-smoke
make supplement
make install-supplement
make supplement-smoke-all
make table1-pilot MODE=rolora
make table1-pilot-all
make full-local
uv run python scripts/local_suite.py full-local --dry-run
uv run python scripts/prep_glue.py --task mnli --dry-run
```
