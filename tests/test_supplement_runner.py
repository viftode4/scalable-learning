from __future__ import annotations

from pathlib import Path


def test_supplement_runner_accepts_config_override() -> None:
    script = Path("scripts/smoke_supplement.sh").read_text()

    assert "CONFIG=${CONFIG:-" in script
    assert "experiments/configs/smoke_supplement.yaml" in script


def test_makefile_has_table1_local_pilot_targets() -> None:
    makefile = Path("Makefile").read_text()

    assert "table1-pilot:" in makefile
    assert "table1-pilot-all:" in makefile
    assert "experiments/configs/table1_local_pilot.yaml" in makefile


def test_makefile_has_medium_pilot_and_summary_targets() -> None:
    makefile = Path("Makefile").read_text()

    assert "table1-medium:" in makefile
    assert "table1-medium-all:" in makefile
    assert "table1-medium-summary:" in makefile
    assert "experiments/configs/table1_local_medium.yaml" in makefile


def test_supplement_runner_stamps_manifest_header() -> None:
    script = Path("scripts/smoke_supplement.sh").read_text()

    assert "# git_sha:" in script
    assert "# config:" in script
    assert "# mode:" in script


def test_makefile_has_feasibility_and_diagnostics_targets() -> None:
    makefile = Path("Makefile").read_text()

    assert "roberta-large-feasibility:" in makefile
    assert "experiments/configs/roberta_large_feasibility.yaml" in makefile
    assert "diagnostics-summary:" in makefile
    assert "--diagnostics" in makefile


def test_roberta_large_feasibility_config_is_supplement_consumable() -> None:
    config = Path("experiments/configs/roberta_large_feasibility.yaml").read_text()

    assert "roberta-large@huggingface_llm" in config
    assert "total_round_num: 1" in config
    assert "client_num: 3" in config
    assert "qnli.json@llm" in config
