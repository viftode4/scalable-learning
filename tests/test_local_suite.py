from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_suite  # noqa: E402


def test_full_local_suite_is_the_expected_local_evidence_chain() -> None:
    steps = local_suite.plan("full-local")

    assert [step.name for step in steps] == [
        "first-party tests + lint",
        "full MNIST Figure-2 run",
        "supplement three-mode smoke",
    ]
    assert [step.command for step in steps] == [
        ["make", "check"],
        ["make", "mnist-paper"],
        ["make", "supplement-smoke-all"],
    ]


def test_dry_run_prints_copy_pastable_commands(capsys) -> None:
    rc = local_suite.main(["full-local", "--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "make check" in out
    assert "make mnist-paper" in out
    assert "make supplement-smoke-all" in out


def test_unknown_suite_is_rejected() -> None:
    try:
        local_suite.plan("cluster")
    except ValueError as exc:
        assert "unknown suite" in str(exc)
    else:
        raise AssertionError("unknown suite should fail")
