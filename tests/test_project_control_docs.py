from __future__ import annotations

from pathlib import Path


def test_progress_has_claim_ledger_and_fallback_triggers() -> None:
    progress = Path("docs/progress.md").read_text()

    assert "## Fallback triggers" in progress
    assert "## Claim ledger" in progress
    assert "C3" in progress
    assert "RoBERTa-Large feasibility" in progress


def test_experiment_matrix_has_dataset_rule_and_comparability_constraints() -> None:
    matrix = Path("docs/experiment-matrix.md").read_text()

    assert "## Dataset rule" in matrix
    assert "MNLI" in matrix
    assert "QNLI" in matrix
    assert "## Comparability constraints" in matrix
    assert "active factor only" in matrix


def test_report_has_claim_led_skeleton() -> None:
    report = Path("report/README.md").read_text()

    assert "## One-sentence thesis" in report
    assert "## Figure and table placeholders" in report
    assert "C5" in report
