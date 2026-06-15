"""Small phase-controller helpers for RoLoRA experiments.

Default RoLoRA alternates B/A/B/A. The helpers here expose opt-in schedules
for low-cost ablations without changing the default behavior.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

VALID_PHASES = frozenset({"A", "B"})
ADAPTIVE_POLICIES = frozenset({"adaptive_refresh", "adaptive_basis_refresh", "abr"})


def normalise_phase_pattern(pattern: str | None) -> str | None:
    """Return an uppercase A/B pattern or None when unset.

    Accepts compact patterns (``BBA``) and comma/space separated patterns
    (``B,B,A``). Raises on invalid characters so a misspelled experiment fails
    early instead of silently running vanilla RoLoRA.
    """
    if pattern is None or pattern.strip() == "":
        return None
    cleaned = pattern.replace(",", " ").replace(";", " ").split()
    compact = "".join(cleaned).upper() if cleaned else pattern.upper()
    if not compact:
        return None
    invalid = sorted(set(compact) - VALID_PHASES)
    if invalid:
        raise ValueError(
            f"SLS_PHASE_PATTERN must contain only A/B phases; got {pattern!r}"
        )
    return compact


def normalise_phase_policy(policy: str | None) -> str | None:
    if policy is None or policy.strip() == "":
        return None
    normalised = policy.strip().lower().replace("-", "_")
    if normalised in {"default", "none", "off", "false", "0"}:
        return None
    if normalised not in ADAPTIVE_POLICIES:
        raise ValueError(
            "SLS_PHASE_POLICY must be one of "
            f"{sorted(ADAPTIVE_POLICIES)}; got {policy!r}"
        )
    return "adaptive_refresh"


def parse_b_warmup_rounds(value: str | int | None) -> int:
    if value is None or value == "":
        return 0
    rounds = int(value)
    if rounds < 0:
        raise ValueError("SLS_B_WARMUP_ROUNDS must be non-negative")
    return rounds


def parse_positive_int(value: str | int | None, default: int, name: str) -> int:
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def parse_float(value: str | float | None, default: float, name: str) -> float:
    if value is None or value == "":
        return default
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _state_file(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    explicit = env.get("SLS_PHASE_STATE_FILE")
    if explicit:
        return Path(explicit)
    run_tag = env.get("SLS_RUN_TAG") or env.get("WANDB_NAME") or str(os.getpid())
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_tag)
    return Path("results") / f"sls_phase_state_{safe}.json"


def load_phase_state(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    path = _state_file(environ)
    try:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text())
        return raw if isinstance(raw, dict) else {}
    except Exception:
        # Phase state is diagnostic/control metadata. A corrupt file should not
        # crash training; the controller falls back to its safe initial phase.
        return {}


def save_phase_state(
    state: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> None:
    path = _state_file(environ)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(state), sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def adaptive_refresh_phase_for_round(
    round_id: int,
    *,
    state: Mapping[str, Any] | None = None,
    min_b_rounds: int = 2,
    max_b_rounds: int = 4,
    val_gain_epsilon: float = 0.001,
) -> str:
    """Choose A/B for event-triggered basis-refresh RoLoRA.

    Interpretation:
    - B rounds fit coefficients on the current A basis.
    - A rounds refresh the basis once B-side progress has stalled.

    The controller is intentionally conservative: it starts with B, enforces a
    minimum B streak, forces an A refresh at max streak, and otherwise refreshes
    A when the previous B round produced too little validation gain.
    """
    if round_id < 0:
        raise ValueError("round_id must be non-negative")
    if min_b_rounds > max_b_rounds:
        raise ValueError("min_b_rounds must be <= max_b_rounds")
    if round_id == 0:
        return "B"

    state = state or {}
    last_phase = str(state.get("last_phase", "")).upper()
    if last_phase == "A":
        return "B"

    b_streak = int(state.get("b_streak", 0))
    if last_phase != "B":
        return "B"
    if b_streak < min_b_rounds:
        return "B"
    if b_streak >= max_b_rounds:
        return "A"

    val_gain = state.get("last_val_gain")
    if val_gain is not None and float(val_gain) <= val_gain_epsilon:
        return "A"

    return "B"


def phase_for_round(
    round_id: int,
    *,
    pattern: str | None = None,
    b_warmup_rounds: str | int | None = None,
    policy: str | None = None,
    state: Mapping[str, Any] | None = None,
    min_b_rounds: str | int | None = None,
    max_b_rounds: str | int | None = None,
    val_gain_epsilon: str | float | None = None,
) -> str:
    """Choose the LoRA factor to train for a communication round.

    Precedence:
    1. ``pattern`` repeats exactly, e.g. ``BBA`` -> B/B/A/B/B/A.
    2. ``policy=adaptive_refresh`` uses validation-gated B-streak control.
    3. ``b_warmup_rounds`` trains B for N initial rounds, then alternates A/B.
    4. default RoLoRA schedule B/A/B/A.
    """
    if round_id < 0:
        raise ValueError("round_id must be non-negative")

    compact = normalise_phase_pattern(pattern)
    if compact is not None:
        return compact[round_id % len(compact)]

    selected_policy = normalise_phase_policy(policy)
    if selected_policy == "adaptive_refresh":
        min_b = parse_positive_int(
            min_b_rounds, 2, "SLS_ADAPTIVE_MIN_B_ROUNDS"
        )
        max_b = parse_positive_int(
            max_b_rounds, 4, "SLS_ADAPTIVE_MAX_B_ROUNDS"
        )
        epsilon = parse_float(
            val_gain_epsilon, 0.001, "SLS_ADAPTIVE_VAL_GAIN_EPSILON"
        )
        return adaptive_refresh_phase_for_round(
            round_id,
            state=state,
            min_b_rounds=min_b,
            max_b_rounds=max_b,
            val_gain_epsilon=epsilon,
        )

    warmup_rounds = parse_b_warmup_rounds(b_warmup_rounds)
    if warmup_rounds and round_id < warmup_rounds:
        return "B"
    if warmup_rounds:
        return "A" if (round_id - warmup_rounds) % 2 == 0 else "B"

    return "B" if round_id % 2 == 0 else "A"


def phase_for_round_from_env(
    round_id: int,
    environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    policy = env.get("SLS_PHASE_POLICY")
    state = load_phase_state(env) if normalise_phase_policy(policy) else None
    return phase_for_round(
        round_id,
        pattern=env.get("SLS_PHASE_PATTERN"),
        b_warmup_rounds=env.get("SLS_B_WARMUP_ROUNDS"),
        policy=policy,
        state=state,
        min_b_rounds=env.get("SLS_ADAPTIVE_MIN_B_ROUNDS"),
        max_b_rounds=env.get("SLS_ADAPTIVE_MAX_B_ROUNDS"),
        val_gain_epsilon=env.get("SLS_ADAPTIVE_VAL_GAIN_EPSILON"),
    )


def update_adaptive_state_after_server_eval(
    round_id: int,
    phase: str,
    metrics: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Persist state used by the next adaptive-refresh phase decision."""
    env = os.environ if environ is None else environ
    if normalise_phase_policy(env.get("SLS_PHASE_POLICY")) != "adaptive_refresh":
        return {}

    previous = load_phase_state(env)
    val_acc = metrics.get("val_acc")
    test_acc = metrics.get("test_acc")
    val_acc_f = float(val_acc) if isinstance(val_acc, (int, float)) else None
    previous_val = previous.get("last_val_acc")
    val_gain = None
    if val_acc_f is not None and isinstance(previous_val, (int, float)):
        val_gain = val_acc_f - float(previous_val)

    phase = phase.upper()
    prev_b_streak = int(previous.get("b_streak", 0))
    b_streak = prev_b_streak + 1 if phase == "B" else 0
    state: dict[str, Any] = {
        "policy": "adaptive_refresh",
        "last_round": int(round_id),
        "last_phase": phase,
        "b_streak": b_streak,
        "last_val_acc": val_acc_f,
        "last_test_acc": float(test_acc) if isinstance(test_acc, (int, float)) else None,
        "last_val_gain": val_gain,
    }
    save_phase_state(state, env)
    return state


def describe_phase_controller(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    pattern = normalise_phase_pattern(env.get("SLS_PHASE_PATTERN"))
    if pattern is not None:
        return f"pattern={pattern}"
    policy = normalise_phase_policy(env.get("SLS_PHASE_POLICY"))
    if policy == "adaptive_refresh":
        min_b = parse_positive_int(
            env.get("SLS_ADAPTIVE_MIN_B_ROUNDS"), 2,
            "SLS_ADAPTIVE_MIN_B_ROUNDS"
        )
        max_b = parse_positive_int(
            env.get("SLS_ADAPTIVE_MAX_B_ROUNDS"), 4,
            "SLS_ADAPTIVE_MAX_B_ROUNDS"
        )
        epsilon = parse_float(
            env.get("SLS_ADAPTIVE_VAL_GAIN_EPSILON"), 0.001,
            "SLS_ADAPTIVE_VAL_GAIN_EPSILON"
        )
        return (
            "adaptive_refresh"
            f"(min_b={min_b},max_b={max_b},val_eps={epsilon:g})"
        )
    warmup_rounds = parse_b_warmup_rounds(env.get("SLS_B_WARMUP_ROUNDS"))
    if warmup_rounds:
        return f"b_warmup_rounds={warmup_rounds}"
    return "default_BA"


# Which round index drives the A/B phase decision. The default ("step_count")
# uses the trainer's local training counter, which only advances when a client
# is sampled -- under partial participation (sample_client_rate < 1) two
# clients in the SAME global round then hold different counters and freeze
# different factors, so the server averages a mix of A- and B-updates and the
# RoLoRA exactness argument (paper Eqs. 3-4) breaks. Opting into "global" makes
# the phase track the global communication round (the paper's odd/even
# definition), so every sampled client agrees on the phase. Default preserves
# byte-for-byte the behaviour of all prior full-participation runs.
_GLOBAL_ROUND_SOURCES = {
    "global", "global_round", "round", "server", "state", "comm", "comm_round",
}


def phase_round_source_is_global(
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get("SLS_PHASE_ROUND_SOURCE")
    if raw is None:
        return False
    return str(raw).strip().lower() in _GLOBAL_ROUND_SOURCES


def resolve_phase_round(
    step_count: int,
    global_round: int | None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Return the round index the phase decision should use.

    ``step_count`` (default) keeps every prior result identical. ``global``
    source returns ``global_round`` so all clients sampled in the same round
    resolve to the same phase; it falls back to ``step_count`` when the global
    round is unavailable (e.g. eval-only calls before the client sets it).
    """
    if phase_round_source_is_global(environ) and global_round is not None:
        return int(global_round)
    return int(step_count)
