"""Run the repo's local evidence suites.

The important distinction:
- ``smoke`` is fast and catches setup breakage.
- ``full-local`` is the strongest laptop-feasible evidence chain: first-party tests,
  the full MNIST Figure-2 sanity run, and the three supplement harness modes.

Full RoBERTa-Large paper reproduction is intentionally not here; that belongs on a
GPU/cluster once DelftBlue/DAIC access is available.
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]


SUITES: dict[str, list[Step]] = {
    "smoke": [
        Step("first-party tests + lint", ["make", "check"]),
        Step("fast MNIST smoke", ["make", "mnist-smoke"]),
        Step("supplement three-mode smoke", ["make", "supplement-smoke-all"]),
    ],
    "full-local": [
        Step("first-party tests + lint", ["make", "check"]),
        Step("full MNIST Figure-2 run", ["make", "mnist-paper"]),
        Step("supplement three-mode smoke", ["make", "supplement-smoke-all"]),
    ],
}


def plan(suite: str) -> list[Step]:
    """Return the ordered commands for a local evidence suite."""
    try:
        return SUITES[suite]
    except KeyError as exc:
        known = ", ".join(sorted(SUITES))
        raise ValueError(f"unknown suite '{suite}' (expected one of: {known})") from exc


def _format(command: Sequence[str]) -> str:
    return " ".join(command)


def run(steps: list[Step], *, dry_run: bool) -> int:
    for number, step in enumerate(steps, start=1):
        print(f"[{number}/{len(steps)}] {step.name}: {_format(step.command)}", flush=True)
        if dry_run:
            continue
        subprocess.run(step.command, check=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=sorted(SUITES), help="local evidence suite to run")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    args = parser.parse_args(argv)
    return run(plan(args.suite), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
