"""Download and tokenize GLUE tasks once, cache them on disk.

The deep-research plan flagged re-tokenizing GLUE on every job as a 30-minute waste.
This script runs once per dataset, writes Arrow shards under ``$SLS_DATA_DIR``
(default ``./data`` locally, ``/scratch/$USER/sls-data`` on the cluster), and prints
a SHA256 manifest so reproducibility is easy to audit.

Usage::

    uv run python scripts/prep_glue.py --task mnli
    uv run python scripts/prep_glue.py --task mnli --dry-run   # just print what would happen
    uv run python scripts/prep_glue.py --task all              # mnli, qqp, qnli

The script is idempotent: if a cached copy already exists, it prints sizes and exits.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

TASKS = {
    "mnli": ("glue", "mnli", "premise", "hypothesis", 3),
    "qqp": ("glue", "qqp", "question1", "question2", 2),
    "qnli": ("glue", "qnli", "question", "sentence", 2),
}


def _sha256_of_dir(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(path).as_posix().encode())
            h.update(f.stat().st_size.to_bytes(8, "little"))
    return h.hexdigest()[:16]


def prep_one(task: str, *, cache_dir: Path, model: str, max_len: int, dry_run: bool) -> None:
    name, config, field_a, field_b, _ = TASKS[task]
    out = cache_dir / f"{task}-roberta-large-{max_len}"
    if out.exists():
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        print(f"[skip] {task}: already cached at {out} ({size / 1e6:.1f} MB)")
        print(f"       sha256[:16]={_sha256_of_dir(out)}")
        return

    if dry_run:
        print(f"[dry-run] would download {name}/{config} and tokenize with {model} (max_len={max_len})")
        print(f"           target: {out}")
        return

    # Imports here so --dry-run works without a full env.
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[prep] {task} → {out}")
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model)
    ds = load_dataset(name, config)

    def tokenize(batch):
        return tok(
            batch[field_a],
            batch[field_b],
            truncation=True,
            max_length=max_len,
            padding=False,
        )

    ds = ds.map(tokenize, batched=True, remove_columns=[field_a, field_b])
    ds.save_to_disk(out.as_posix())

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"[done] {task}: {size / 1e6:.1f} MB at {out}")
    print(f"       sha256[:16]={_sha256_of_dir(out)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="mnli", choices=[*TASKS.keys(), "all"])
    p.add_argument("--model", default="roberta-large")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("SLS_DATA_DIR", "./data")),
    )
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"cache_dir={args.cache_dir}")
    tasks = list(TASKS) if args.task == "all" else [args.task]
    for t in tasks:
        prep_one(
            t,
            cache_dir=args.cache_dir,
            model=args.model,
            max_len=args.max_len,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
