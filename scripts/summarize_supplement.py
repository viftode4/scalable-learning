"""Summarize FederatedScope supplement logs into a small evidence table."""

from __future__ import annotations

import argparse
import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

RESULT_RE = re.compile(r"\{.*'Results_raw':.*\}")


@dataclass(frozen=True)
class Result:
    round: int
    test_acc: float
    val_acc: float


@dataclass(frozen=True)
class Row:
    mode: str
    round: int
    test_acc: float
    val_acc: float
    marker: str
    log: Path


def parse_result_line(line: str) -> Result | None:
    """Parse one FederatedScope ``Results_raw`` log line."""
    match = RESULT_RE.search(line)
    if not match:
        return None
    try:
        payload = ast.literal_eval(match.group(0))
    except (SyntaxError, ValueError):
        return None
    metrics = payload["Results_raw"]
    if "test_acc" not in metrics or "val_acc" not in metrics:
        return None
    return Result(
        round=int(payload["Round"]),
        test_acc=float(metrics["test_acc"]),
        val_acc=float(metrics["val_acc"]),
    )


def _mode_from_name(path: Path, prefix: str) -> str:
    stem = path.stem
    return stem.removeprefix(f"{prefix}_")


def summarize_log(path: Path, prefix: str) -> Row:
    marker = ""
    result: Result | None = None
    for line in path.read_text(errors="replace").splitlines():
        if "[sls-rolora]" in line:
            marker = line.strip()
        parsed = parse_result_line(line)
        if parsed is not None:
            result = parsed
    if result is None:
        raise ValueError(f"no Results_raw metrics found in {path}")
    if not marker:
        raise ValueError(f"no [sls-rolora] patch marker found in {path}")
    return Row(
        mode=_mode_from_name(path, prefix),
        round=result.round,
        test_acc=result.test_acc,
        val_acc=result.val_acc,
        marker=marker,
        log=path,
    )


def summarize_logs(results_dir: Path, prefix: str) -> list[Row]:
    logs = sorted(results_dir.glob(f"{prefix}_*.log"))
    if not logs:
        raise FileNotFoundError(f"no logs found for prefix '{prefix}' in {results_dir}")
    return [summarize_log(path, prefix) for path in logs]


def to_markdown(rows: list[Row]) -> str:
    lines = [
        "| mode | round | test_acc | val_acc | marker |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.mode} | {row.round} | {row.test_acc:.6f} | "
            f"{row.val_acc:.6f} | {row.marker} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="table1_pilot", help="log filename prefix")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    print(to_markdown(summarize_logs(args.results_dir, args.prefix)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
