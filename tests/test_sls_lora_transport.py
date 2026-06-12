from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "code/harness/rolora-supplement/RoLoRA-code/federatedscope/core/"
    "sls_lora_transport.py"
)


def load_transport_module():
    spec = importlib.util.spec_from_file_location(
        "sls_lora_transport", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _pair_state(a: torch.Tensor, b: torch.Tensor) -> dict:
    return {
        "encoder.layer.0.attn.lora_A.default.weight": a.clone(),
        "encoder.layer.0.attn.lora_B.default.weight": b.clone(),
    }


A_KEY = "encoder.layer.0.attn.lora_A.default.weight"
B_KEY = "encoder.layer.0.attn.lora_B.default.weight"


def test_transport_exact_when_new_basis_spans_old() -> None:
    transport = load_transport_module()
    torch.manual_seed(7)
    a_old = torch.randn(4, 9)
    b_old = torch.randn(7, 4)
    rotation = torch.randn(4, 4) + 4.0 * torch.eye(4)
    a_new = rotation @ a_old

    old_state = _pair_state(a_old, b_old)
    new_state = _pair_state(a_new, b_old)
    stats = transport.apply_basis_transport(old_state, new_state)

    b_after = new_state[B_KEY]
    assert torch.allclose(
        b_after @ a_new, b_old @ a_old, atol=1e-4, rtol=1e-4
    )
    assert stats["transport_pair_count"] == 1
    assert stats["transport_max_function_rel_error"] < 1e-4
    # A itself must not be touched: transport only re-expresses B.
    assert torch.equal(new_state[A_KEY], a_new)


def test_transport_matches_pinv_reference_for_general_basis_change() -> None:
    transport = load_transport_module()
    torch.manual_seed(11)
    a_old = torch.randn(3, 8)
    b_old = torch.randn(5, 3)
    a_new = torch.randn(3, 8)

    old_state = _pair_state(a_old, b_old)
    new_state = _pair_state(a_new, b_old)
    transport.apply_basis_transport(old_state, new_state)

    reference = b_old @ a_old @ torch.linalg.pinv(a_new)
    assert torch.allclose(new_state[B_KEY], reference, atol=1e-4, rtol=1e-4)


def test_transport_noop_when_basis_unchanged() -> None:
    transport = load_transport_module()
    torch.manual_seed(3)
    a = torch.randn(2, 6)
    b = torch.randn(4, 2)

    old_state = _pair_state(a, b)
    new_state = _pair_state(a, b)
    stats = transport.apply_basis_transport(old_state, new_state)

    assert stats["transport_pair_count"] == 0
    assert stats["transport_unchanged_count"] == 1
    assert torch.equal(new_state[B_KEY], b)


def test_transport_beta_damps_towards_identity() -> None:
    transport = load_transport_module()
    torch.manual_seed(5)
    a_old = torch.randn(3, 7)
    b_old = torch.randn(4, 3)
    rotation = torch.randn(3, 3) + 3.0 * torch.eye(3)
    a_new = rotation @ a_old

    full_state = _pair_state(a_new, b_old)
    transport.apply_basis_transport(_pair_state(a_old, b_old), full_state)
    b_full = full_state[B_KEY]

    frozen_state = _pair_state(a_new, b_old)
    stats = transport.apply_basis_transport(
        _pair_state(a_old, b_old), frozen_state, beta=0.0
    )
    assert torch.allclose(frozen_state[B_KEY], b_old, atol=1e-6)
    assert stats["transport_pair_count"] == 1

    half_state = _pair_state(a_new, b_old)
    transport.apply_basis_transport(
        _pair_state(a_old, b_old), half_state, beta=0.5
    )
    assert torch.allclose(
        half_state[B_KEY], 0.5 * b_old + 0.5 * b_full, atol=1e-4, rtol=1e-4
    )


def test_transport_falls_back_to_identity_on_degenerate_new_basis() -> None:
    transport = load_transport_module()
    torch.manual_seed(9)
    a_old = torch.randn(3, 7)
    b_old = torch.randn(4, 3)
    a_new = torch.zeros(3, 7)

    new_state = _pair_state(a_new, b_old)
    stats = transport.apply_basis_transport(
        _pair_state(a_old, b_old), new_state
    )

    assert stats["transport_fallback_count"] == 1
    assert torch.equal(new_state[B_KEY], b_old)


def test_transport_from_env_is_disabled_by_default() -> None:
    transport = load_transport_module()
    a_old = torch.randn(2, 5)
    b_old = torch.randn(3, 2)
    a_new = torch.randn(2, 5)
    new_state = _pair_state(a_new, b_old)

    stats = transport.apply_lora_transport_from_env(
        _pair_state(a_old, b_old), new_state, environ={}
    )

    assert stats == {}
    assert torch.equal(new_state[B_KEY], b_old)


def test_transport_env_aliases_and_unknown_values() -> None:
    transport = load_transport_module()
    for alias in ("ls", "least_squares", "transport", "basis_transport"):
        assert transport.normalise_lora_transport(alias) == "ls"
    for disabled in (None, "", "0", "off", "none"):
        assert transport.normalise_lora_transport(disabled) is None
    try:
        transport.normalise_lora_transport("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown transport value must raise ValueError")


class DummyServerModel:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


def test_aggregator_transports_b_after_partial_a_round(monkeypatch) -> None:
    sys.path.insert(0, str(ROOT / "code/harness/rolora-supplement/RoLoRA-code"))
    from federatedscope.core.aggregators.clients_avg_aggregator import (  # noqa: E402
        ClientsAvgAggregator,
    )

    cfg = SimpleNamespace(
        federate=SimpleNamespace(ignore_weight=False, use_ss=False),
    )
    torch.manual_seed(17)
    server_a = torch.randn(2, 6)
    server_b = torch.randn(4, 2)
    # Clients return A in the same row space as the old basis, so the
    # transported function must be preserved exactly.
    r1 = torch.tensor([[1.5, 0.25], [-0.5, 2.0]])
    r2 = torch.tensor([[2.0, -0.25], [0.75, 1.25]])
    client_a1 = r1 @ server_a
    client_a2 = r2 @ server_a
    avg_a = (client_a1 + client_a2) / 2

    model = DummyServerModel(
        {
            "block.lora_A.default.weight": server_a.clone(),
            "block.lora_B.default.weight": server_b.clone(),
        }
    )
    aggregator = ClientsAvgAggregator(model=model, config=cfg)

    monkeypatch.setenv("SLS_LORA_TRANSPORT", "ls")
    monkeypatch.delenv("SLS_LORA_GAUGE", raising=False)
    result = aggregator.aggregate(
        {
            "round": 1,
            "client_feedback": [
                (1, {"block.lora_A.default.weight": client_a1.clone()}),
                (1, {"block.lora_A.default.weight": client_a2.clone()}),
            ],
        }
    )

    assert "block.lora_B.default.weight" in result
    a_after = result["block.lora_A.default.weight"]
    b_after = result["block.lora_B.default.weight"]
    assert torch.allclose(a_after, avg_a, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        b_after @ a_after, server_b @ server_a, atol=1e-4, rtol=1e-4
    )


def test_aggregator_transport_is_noop_on_b_round(monkeypatch) -> None:
    sys.path.insert(0, str(ROOT / "code/harness/rolora-supplement/RoLoRA-code"))
    from federatedscope.core.aggregators.clients_avg_aggregator import (  # noqa: E402
        ClientsAvgAggregator,
    )

    cfg = SimpleNamespace(
        federate=SimpleNamespace(ignore_weight=False, use_ss=False),
    )
    server_a = torch.randn(2, 6)
    server_b = torch.randn(4, 2)
    client_b = torch.randn(4, 2)
    model = DummyServerModel(
        {
            "block.lora_A.default.weight": server_a.clone(),
            "block.lora_B.default.weight": server_b.clone(),
        }
    )
    aggregator = ClientsAvgAggregator(model=model, config=cfg)

    monkeypatch.setenv("SLS_LORA_TRANSPORT", "ls")
    monkeypatch.delenv("SLS_LORA_GAUGE", raising=False)
    result = aggregator.aggregate(
        {
            "round": 2,
            "client_feedback": [
                (1, {"block.lora_B.default.weight": client_b.clone()}),
            ],
        }
    )

    # B-rounds leave A untouched, so transport must not modify the payload.
    assert torch.allclose(
        result["block.lora_B.default.weight"], client_b, atol=1e-6
    )
    assert "block.lora_A.default.weight" not in result


def test_transport_composes_with_orthogonal_gauge(monkeypatch) -> None:
    sys.path.insert(0, str(ROOT / "code/harness/rolora-supplement/RoLoRA-code"))
    from federatedscope.core.aggregators.clients_avg_aggregator import (  # noqa: E402
        ClientsAvgAggregator,
    )

    cfg = SimpleNamespace(
        federate=SimpleNamespace(ignore_weight=False, use_ss=False),
    )
    torch.manual_seed(23)
    server_a = torch.randn(2, 6)
    server_b = torch.randn(4, 2)
    rotation = torch.tensor([[1.25, 0.5], [-0.25, 1.5]])
    client_a = rotation @ server_a
    model = DummyServerModel(
        {
            "block.lora_A.default.weight": server_a.clone(),
            "block.lora_B.default.weight": server_b.clone(),
        }
    )
    aggregator = ClientsAvgAggregator(model=model, config=cfg)

    monkeypatch.setenv("SLS_LORA_TRANSPORT", "ls")
    monkeypatch.setenv("SLS_LORA_GAUGE", "orthogonal_a")
    result = aggregator.aggregate(
        {
            "round": 3,
            "client_feedback": [
                (1, {"block.lora_A.default.weight": client_a.clone()}),
            ],
        }
    )

    a_after = result["block.lora_A.default.weight"]
    b_after = result["block.lora_B.default.weight"]
    # Transport restores the pre-round function, gauge re-orthonormalizes A
    # while preserving the (restored) product.
    assert torch.allclose(
        b_after @ a_after, server_b @ server_a, atol=1e-4, rtol=1e-4
    )
    assert torch.allclose(
        a_after @ a_after.T, torch.eye(2), atol=1e-5, rtol=1e-5
    )
