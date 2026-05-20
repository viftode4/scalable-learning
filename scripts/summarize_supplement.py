"""Summarize FederatedScope supplement logs into evidence tables.

Default output keeps the compact final-metric table used by the local ledger.
``--diagnostics`` emits a per-result table with manifest fields and the latest
RoLoRA phase marker seen before each metric line. This is intentionally log-only:
it does not require importing the gitignored supplement runtime.
"""

from __future__ import annotations

import argparse
import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

RESULT_RE = re.compile(r"\{.*'Results_raw':.*\}")
ROUND_RE = re.compile(r"\bround\s+(\d+)\b", re.IGNORECASE)
PHASE_RE = re.compile(r"(train\s+(?:A|B|both))\b", re.IGNORECASE)


@dataclass(frozen=True)
class Result:
    round: int
    test_acc: float
    val_acc: float
    test_loss: float | None = None
    val_loss: float | None = None


@dataclass(frozen=True)
class Row:
    mode: str
    round: int
    test_acc: float
    val_acc: float
    marker: str
    log: Path


@dataclass(frozen=True)
class DiagnosticRow:
    mode: str
    config: str
    git_sha: str
    round: int
    marker_round: int | None
    phase: str
    test_acc: float
    val_acc: float
    test_loss: float | None
    val_loss: float | None
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
        test_loss=_optional_float(metrics.get("test_loss")),
        val_loss=_optional_float(metrics.get("val_loss")),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_manifest(lines: Sequence[str]) -> dict[str, str]:
    """Parse ``# key: value`` headers stamped by ``smoke_supplement.sh``."""
    manifest: dict[str, str] = {}
    for line in lines:
        if not line.startswith("# ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        manifest[key.strip()] = value.strip()
    return manifest


def parse_marker(marker: str) -> tuple[int | None, str]:
    """Extract marker round and phase from a ``[sls-rolora]`` line."""
    round_match = ROUND_RE.search(marker)
    phase_match = PHASE_RE.search(marker)
    marker_round = int(round_match.group(1)) if round_match else None
    phase = phase_match.group(1).lower() if phase_match else "unknown"
    return marker_round, phase


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


def summarize_diagnostic_log(path: Path, prefix: str) -> list[DiagnosticRow]:
    """Return one diagnostic row per parsed result line in a log."""
    lines = path.read_text(errors="replace").splitlines()
    manifest = parse_manifest(lines)
    mode = manifest.get("mode", _mode_from_name(path, prefix))
    config = manifest.get("config", "unknown")
    git_sha = manifest.get("git_sha", "unknown")
    markers = [line.strip() for line in lines if "[sls-rolora]" in line]
    marker = ""
    rows: list[DiagnosticRow] = []
    for line in lines:
        if "[sls-rolora]" in line:
            marker = line.strip()
        parsed = parse_result_line(line)
        if parsed is None:
            continue
        effective_marker = marker
        marker_round, phase = parse_marker(effective_marker)
        rows.append(
            DiagnosticRow(
                mode=mode,
                config=config,
                git_sha=git_sha,
                round=parsed.round,
                marker_round=marker_round,
                phase=phase,
                test_acc=parsed.test_acc,
                val_acc=parsed.val_acc,
                test_loss=parsed.test_loss,
                val_loss=parsed.val_loss,
                marker=effective_marker,
                log=path,
            )
        )
    if not rows:
        raise ValueError(f"no Results_raw metrics found in {path}")
    if not markers:
        raise ValueError(f"no [sls-rolora] patch marker found in {path}")
    return rows


def summarize_logs(results_dir: Path, prefix: str) -> list[Row]:
    logs = sorted(results_dir.glob(f"{prefix}_*.log"))
    if not logs:
        raise FileNotFoundError(f"no logs found for prefix '{prefix}' in {results_dir}")
    return [summarize_log(path, prefix) for path in logs]


def summarize_diagnostics(results_dir: Path, prefix: str) -> list[DiagnosticRow]:
    logs = sorted(results_dir.glob(f"{prefix}_*.log"))
    if not logs:
        raise FileNotFoundError(f"no logs found for prefix '{prefix}' in {results_dir}")
    rows: list[DiagnosticRow] = []
    for path in logs:
        rows.extend(summarize_diagnostic_log(path, prefix))
    return rows


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


def diagnostics_to_markdown(rows: list[DiagnosticRow]) -> str:
    lines = [
        "| mode | git_sha | config | round | marker_round | phase | test_acc | val_acc | test_loss | val_loss | log |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.mode} | {row.git_sha} | {row.config} | {row.round} | "
            f"{_format_optional_int(row.marker_round)} | {row.phase} | "
            f"{row.test_acc:.6f} | {row.val_acc:.6f} | "
            f"{_format_optional_float(row.test_loss)} | {_format_optional_float(row.val_loss)} | {row.log} |"
        )
    return "\n".join(lines)


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _format_optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="table1_pilot", help="log filename prefix")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="emit per-result diagnostics with manifest and latest phase marker",
    )
    args = parser.parse_args(argv)
    if args.diagnostics:
        print(diagnostics_to_markdown(summarize_diagnostics(args.results_dir, args.prefix)))
    else:
        print(to_markdown(summarize_logs(args.results_dir, args.prefix)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
