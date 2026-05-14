# Environment setup

The repo uses **uv** for env management. `pyproject.toml` declares deps; `uv.lock` pins exact resolved versions.

## Install uv (one-time)
macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or via Homebrew: `brew install uv`. Linux: same curl one-liner.

## Bring up a fresh env
```bash
git clone <this-repo> && cd scalable-learning
git submodule update --init --recursive
uv sync
```

`uv sync` reads `uv.lock` and materialises `.venv/`. First run downloads ~3 GB of wheels (torch, etc.).

## Sanity-check
```bash
uv run python -c "import torch, peft, transformers; print(torch.__version__, peft.__version__, transformers.__version__)"
```
Expected: torch 2.x, **peft 0.10.0**, transformers 4.4x.

## Common operations
```bash
uv add <pkg>           # adds to pyproject and relocks
uv add --dev <pkg>     # dev-only (pytest, ruff, etc.)
uv lock --upgrade      # bumps every dep to latest within declared bounds
uv run <cmd>           # run a command inside the env without activating it
```

Always commit `pyproject.toml` and `uv.lock` together — keep them in lockstep.

## Env vars we use
| Variable | Default | Meaning |
|---|---|---|
| `SLS_DATA_DIR` | `./data` | Where `scripts/prep_glue.py` caches tokenized GLUE shards. Set to `/scratch/$USER/sls-data` on DelftBlue. |
| `SLS_RESULTS_DIR` | `./results` | Where training output / plots are written. Gitignored. |

## On DelftBlue
uv works without sudo. The Slurm templates `module load` system Python first, then run `uv sync` and `uv run` inside the job. See `docs/setup/delftblue.md` once TA instructions arrive.
