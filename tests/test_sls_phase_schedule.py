from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/trainer/"
    / "sls_phase_schedule.py"
)
spec = importlib.util.spec_from_file_location("sls_phase_schedule", MODULE_PATH)
assert spec is not None and spec.loader is not None
sls_phase_schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sls_phase_schedule)


def phases(n: int, **kwargs) -> list[str]:
    return [sls_phase_schedule.phase_for_round(round_id, **kwargs) for round_id in range(n)]


def test_default_schedule_matches_rolora_ba_alternation() -> None:
    assert phases(6) == ["B", "A", "B", "A", "B", "A"]


def test_pattern_repeats_exactly() -> None:
    assert phases(7, pattern="BBA") == ["B", "B", "A", "B", "B", "A", "B"]


def test_pattern_accepts_separators_and_lowercase() -> None:
    assert phases(4, pattern="b, a; b") == ["B", "A", "B", "B"]


def test_b_warmup_then_alternates_a_b() -> None:
    assert phases(6, b_warmup_rounds=2) == ["B", "B", "A", "B", "A", "B"]


def test_pattern_takes_precedence_over_adaptive_policy() -> None:
    assert phases(4, pattern="AB", policy="adaptive_refresh") == ["A", "B", "A", "B"]


def test_default_policy_string_is_treated_as_unset() -> None:
    assert phases(4, policy="default") == ["B", "A", "B", "A"]
    assert phases(4, pattern="BBA", policy="default") == ["B", "B", "A", "B"]


def test_pattern_takes_precedence_over_warmup() -> None:
    assert phases(4, pattern="AB", b_warmup_rounds=3) == ["A", "B", "A", "B"]


def test_invalid_pattern_fails_fast() -> None:
    with pytest.raises(ValueError, match="only A/B"):
        phases(1, pattern="BC")


def test_negative_warmup_fails_fast() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        phases(1, b_warmup_rounds=-1)


def test_adaptive_refresh_starts_with_b() -> None:
    assert sls_phase_schedule.phase_for_round(0, policy="adaptive_refresh") == "B"


def test_adaptive_refresh_enforces_minimum_b_streak() -> None:
    state = {"last_phase": "B", "b_streak": 1, "last_val_gain": -1.0}
    assert (
        sls_phase_schedule.phase_for_round(
            2, policy="adaptive_refresh", state=state, min_b_rounds=2
        )
        == "B"
    )


def test_adaptive_refresh_switches_to_a_when_b_stalls_after_minimum() -> None:
    state = {"last_phase": "B", "b_streak": 2, "last_val_gain": 0.0001}
    assert (
        sls_phase_schedule.phase_for_round(
            3,
            policy="adaptive_refresh",
            state=state,
            min_b_rounds=2,
            max_b_rounds=4,
            val_gain_epsilon=0.001,
        )
        == "A"
    )


def test_adaptive_refresh_keeps_b_when_b_is_still_improving() -> None:
    state = {"last_phase": "B", "b_streak": 2, "last_val_gain": 0.01}
    assert (
        sls_phase_schedule.phase_for_round(
            3,
            policy="adaptive_refresh",
            state=state,
            min_b_rounds=2,
            max_b_rounds=4,
            val_gain_epsilon=0.001,
        )
        == "B"
    )


def test_adaptive_refresh_forces_a_at_max_b_streak() -> None:
    state = {"last_phase": "B", "b_streak": 4, "last_val_gain": 0.1}
    assert (
        sls_phase_schedule.phase_for_round(
            5,
            policy="adaptive_refresh",
            state=state,
            min_b_rounds=2,
            max_b_rounds=4,
        )
        == "A"
    )


def test_adaptive_refresh_returns_to_b_after_a_refresh() -> None:
    state = {"last_phase": "A", "b_streak": 0, "last_val_gain": -1.0}
    assert sls_phase_schedule.phase_for_round(4, policy="adaptive_refresh", state=state) == "B"


def test_adaptive_state_persists_val_gain_and_b_streak(tmp_path: Path) -> None:
    env = {
        "SLS_PHASE_POLICY": "adaptive_refresh",
        "SLS_PHASE_STATE_FILE": str(tmp_path / "phase.json"),
    }
    first = sls_phase_schedule.update_adaptive_state_after_server_eval(
        0, "B", {"val_acc": 0.50, "test_acc": 0.51}, env
    )
    second = sls_phase_schedule.update_adaptive_state_after_server_eval(
        1, "B", {"val_acc": 0.505, "test_acc": 0.52}, env
    )
    assert first["b_streak"] == 1
    assert second["b_streak"] == 2
    assert second["last_val_gain"] == pytest.approx(0.005)
    assert sls_phase_schedule.load_phase_state(env)["last_round"] == 1


def test_adaptive_phase_from_env_uses_persisted_state(tmp_path: Path) -> None:
    env = {
        "SLS_PHASE_POLICY": "adaptive_refresh",
        "SLS_PHASE_STATE_FILE": str(tmp_path / "phase.json"),
        "SLS_ADAPTIVE_MIN_B_ROUNDS": "2",
        "SLS_ADAPTIVE_MAX_B_ROUNDS": "4",
        "SLS_ADAPTIVE_VAL_GAIN_EPSILON": "0.001",
    }
    sls_phase_schedule.update_adaptive_state_after_server_eval(
        0, "B", {"val_acc": 0.50}, env
    )
    sls_phase_schedule.update_adaptive_state_after_server_eval(
        1, "B", {"val_acc": 0.5005}, env
    )
    assert sls_phase_schedule.phase_for_round_from_env(2, env) == "A"


def test_invalid_adaptive_policy_fails_fast() -> None:
    with pytest.raises(ValueError, match="SLS_PHASE_POLICY"):
        sls_phase_schedule.phase_for_round(0, policy="mystery")
