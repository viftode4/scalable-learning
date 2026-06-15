"""Phase round-source resolution for RoLoRA under partial participation.

The A/B phase must track the GLOBAL communication round when
SLS_PHASE_ROUND_SOURCE=global, so that two clients sampled in the same round
(but with different local step_counts, because they sat out different rounds)
agree on which factor to freeze. The default path keeps using the local
step_count so every prior full-participation result is byte-for-byte unchanged.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/trainer/"
    "sls_phase_schedule.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "sls_phase_schedule", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = load_module()


# --- source detection -------------------------------------------------------

def test_default_source_is_not_global():
    assert M.phase_round_source_is_global({}) is False


def test_global_aliases_detected():
    for val in ["global", "round", "server", "state", "GLOBAL", " Global "]:
        assert M.phase_round_source_is_global(
            {"SLS_PHASE_ROUND_SOURCE": val}
        ) is True


def test_explicit_step_count_source_is_not_global():
    assert M.phase_round_source_is_global(
        {"SLS_PHASE_ROUND_SOURCE": "step_count"}
    ) is False


# --- round resolution -------------------------------------------------------

def test_default_resolves_to_step_count():
    # global round present but env unset -> step_count wins (prior behaviour).
    assert M.resolve_phase_round(5, 3, {}) == 5


def test_global_resolves_to_global_round():
    env = {"SLS_PHASE_ROUND_SOURCE": "global"}
    assert M.resolve_phase_round(5, 3, env) == 3


def test_global_falls_back_when_round_missing():
    # eval-only calls before the client sets the round must not crash.
    env = {"SLS_PHASE_ROUND_SOURCE": "global"}
    assert M.resolve_phase_round(7, None, env) == 7


# --- the property that matters (P3): cohort phase agreement -----------------

def test_partial_participation_cohort_agrees_on_phase_under_global():
    """Two clients in the same global round, different step_counts.

    Under the global source they must resolve to the SAME phase; under the
    default step_count source they diverge -- which is exactly the
    desynchronization bug that reintroduces aggregation interference.
    """
    env = {"SLS_PHASE_ROUND_SOURCE": "global"}
    global_round = 4
    # client_a was sampled often (step_count high), client_b sat out rounds.
    step_a, step_b = 6, 3

    phase_a = M.phase_for_round_from_env(
        M.resolve_phase_round(step_a, global_round, env), env
    )
    phase_b = M.phase_for_round_from_env(
        M.resolve_phase_round(step_b, global_round, env), env
    )
    assert phase_a == phase_b, "global source must synchronize the cohort"

    # Default source: the same two clients disagree (documents the bug).
    default_env: dict = {}
    phase_a_def = M.phase_for_round_from_env(
        M.resolve_phase_round(step_a, global_round, default_env), default_env
    )
    phase_b_def = M.phase_for_round_from_env(
        M.resolve_phase_round(step_b, global_round, default_env), default_env
    )
    # step 6 -> B (even), step 3 -> A (odd): they diverge.
    assert phase_a_def != phase_b_def


def test_global_source_matches_paper_odd_even():
    """Global round drives the paper's odd/even B/A schedule regardless of
    how many rounds a given client actually trained."""
    env = {"SLS_PHASE_ROUND_SOURCE": "global"}
    # default pattern: even round -> B, odd round -> A
    for rnd, expected in [(0, "B"), (1, "A"), (2, "B"), (3, "A")]:
        for any_step in [0, 1, 99]:
            got = M.phase_for_round_from_env(
                M.resolve_phase_round(any_step, rnd, env), env
            )
            assert got == expected, (rnd, any_step, got)
