# Code

Our Python code, the harness checkouts, and the configs that drive them.

## Layout
```
code/
  harness/
    fedsa-lora/          # Git submodule: our fork of Pengxin-Guo/FedSA-LoRA
    rolora-supplement/   # OpenReview supplementary zip extract — gitignored, user-fetched
    README.md
```

The Python package itself (`sls_rolora`) lands here in the next iteration once we have a federated training entrypoint. For now this directory holds the harness checkouts only.

## Environment setup
The repo is uv-managed (`pyproject.toml` + `uv.lock`).

```bash
# Install uv once: https://github.com/astral-sh/uv (curl install or brew)
uv sync                              # creates .venv/ and installs pinned deps
uv run python -c "import torch, peft; print(torch.__version__, peft.__version__)"
```

Expected output: `torch 2.x.x 0.10.0` (peft pinned at 0.10.0 — see ADR 0002).

## Adding / updating deps
```bash
uv add <package>                     # adds + relocks
uv add --dev <package>               # dev-only
uv lock --upgrade                    # bumps everything to latest compatible
```

Always commit `pyproject.toml` and `uv.lock` together.

## Running things
- MNIST Figure-2: `uv run python notebooks/mnist_fig2.py`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`

## On the cluster
See `docs/setup/delftblue.md` (currently waiting on TA instructions). uv works without sudo and produces a hermetic env, so the local workflow should carry over.
