# ADR 0002 — Python environment via uv

**Status:** Accepted (2026-05-14)

## Context

We need a reproducible Python environment for three contributors across macOS laptops and the DelftBlue cluster. Constraints:

- DelftBlue compute nodes do not allow sudo.
- The deep-research plan flagged PEFT version drift as a real bug source (`peft==0.7` vs `peft==0.10` differ in how `lora_alpha` interacts with `r`; one update silently changed Init[B]). Pinning is mandatory.
- The default-low-priority job queue makes fast iteration valuable — env creation should not eat minutes per fresh clone.

## Decision

Use **uv** (`astral-sh/uv`) for env management.

- Source of truth: `pyproject.toml` (declarations) + `uv.lock` (resolved hashes).
- Python pinned to `3.11` via `.python-version`.
- Core deps pinned: `torch>=2.1,<2.4`, `transformers>=4.40,<4.50`, **`peft==0.10.0`**, `datasets>=2.18`, `accelerate>=0.30`, `numpy>=1.26,<2.0`.
- Dev deps in `[dependency-groups.dev]`: `pytest`, `ruff`, `pytest-xdist`.
- `.venv/` gitignored — created locally per machine.

## Consequences

- One command (`uv sync`) brings any machine to a byte-identical env (modulo wheels for the OS/arch).
- No conda dependency — works on DelftBlue without modules beyond a recent Python.
- Lockfile churn on dep changes is concentrated in `uv.lock`; reviews stay clean.
- If a teammate prefers conda/pip, they can read `pyproject.toml` and recreate manually; we don't ship a `requirements.txt`.

## Alternatives rejected

- **conda/mamba** — heavy on disk, slower env creation, brings unnecessary CUDA stack on macOS.
- **plain venv + requirements.txt** — no lockfile means resolution drift; the peft pinning issue would not be enforced.
- **Poetry** — slower, more opinionated, and uv covers the same ground with less ceremony.
