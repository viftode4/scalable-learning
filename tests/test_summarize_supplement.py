from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_supplement  # noqa: E402


def test_parse_result_line_extracts_metrics() -> None:
    line = "2026-05-14 INFO: {'Role': 'Client #3', 'Round': 3, 'Results_raw': {'test_acc': 0.511258, 'val_acc': 0.504298, 'test_loss': 1.2}}"

    result = summarize_supplement.parse_result_line(line)

    assert result is not None
    assert result.round == 3
    assert result.test_acc == 0.511258
    assert result.val_acc == 0.504298


def test_parse_result_line_skips_incomplete_metrics() -> None:
    line = "INFO: {'Role': 'Server', 'Round': 3, 'Results_raw': {'train_loss': 1.0}}"

    assert summarize_supplement.parse_result_line(line) is None


def test_parse_result_line_skips_non_literal_payloads() -> None:
    line = "INFO: {'Role': 'Server', 'Round': 3, 'Results_raw': {'test_acc': nan, 'val_acc': 0.5}}"

    assert summarize_supplement.parse_result_line(line) is None


def test_summarize_logs_returns_sorted_rows(tmp_path: Path) -> None:
    (tmp_path / "table1_pilot_lora.log").write_text(
        "[sls-rolora] LoRA round 4: train both\n"
        "INFO: {'Role': 'Client #3', 'Round': 3, 'Results_raw': {'test_acc': 0.50, 'val_acc': 0.49}}\n"
    )
    (tmp_path / "table1_pilot_rolora.log").write_text(
        "[sls-rolora] RoLoRA round 4: train B\n"
        "INFO: {'Role': 'Client #3', 'Round': 3, 'Results_raw': {'test_acc': 0.52, 'val_acc': 0.51}}\n"
    )

    rows = summarize_supplement.summarize_logs(tmp_path, "table1_pilot")

    assert [row.mode for row in rows] == ["lora", "rolora"]
    assert rows[1].marker == "[sls-rolora] RoLoRA round 4: train B"


def test_markdown_table_contains_metrics(tmp_path: Path) -> None:
    (tmp_path / "table1_pilot_lora.log").write_text(
        "[sls-rolora] LoRA round 4: train both\n"
        "INFO: {'Role': 'Client #3', 'Round': 3, 'Results_raw': {'test_acc': 0.50, 'val_acc': 0.49}}\n"
    )

    table = summarize_supplement.to_markdown(
        summarize_supplement.summarize_logs(tmp_path, "table1_pilot")
    )

    assert "| mode | round | test_acc | val_acc | marker |" in table
    assert "| lora | 3 | 0.500000 | 0.490000 | [sls-rolora] LoRA round 4: train both |" in table


def test_parse_manifest_extracts_header_fields() -> None:
    lines = [
        "# git_sha: abc123",
        "# config: experiments/configs/table1_local_pilot.yaml",
        "# mode: rolora",
    ]

    manifest = summarize_supplement.parse_manifest(lines)

    assert manifest["git_sha"] == "abc123"
    assert manifest["config"] == "experiments/configs/table1_local_pilot.yaml"
    assert manifest["mode"] == "rolora"


def test_diagnostics_table_contains_manifest_and_phase(tmp_path: Path) -> None:
    (tmp_path / "table1_pilot_rolora.log").write_text(
        "# git_sha: abc123\n"
        "# config: experiments/configs/table1_local_pilot.yaml\n"
        "# mode: rolora\n"
        "[sls-rolora] RoLoRA round 4: train B\n"
        "INFO: {'Role': 'Client #3', 'Round': 3, 'Results_raw': {'test_acc': 0.52, 'val_acc': 0.51, 'test_loss': 1.2, 'val_loss': 1.3}}\n"
    )

    table = summarize_supplement.diagnostics_to_markdown(
        summarize_supplement.summarize_diagnostics(tmp_path, "table1_pilot")
    )

    assert "| mode | git_sha | config | round | marker_round | phase |" in table
    assert "| rolora | abc123 | experiments/configs/table1_local_pilot.yaml | 3 | 4 | train b |" in table
    assert "1.200000" in table
    assert "1.300000" in table
