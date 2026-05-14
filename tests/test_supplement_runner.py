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
