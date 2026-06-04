from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "code/harness/rolora-supplement/RoLoRA-code/federatedscope/core/sls_lora_gauge.py"
)


def load_gauge_module():
    spec = importlib.util.spec_from_file_location("sls_lora_gauge", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_orthogonal_a_gauge_preserves_adapter_product() -> None:
    gauge = load_gauge_module()
    torch.manual_seed(13)
    a = torch.randn(4, 9)
    b = torch.randn(7, 4)
    classifier = torch.randn(3, 7)
    state = {
        "encoder.layer.0.attn.lora_A.default.weight": a.clone(),
        "encoder.layer.0.attn.lora_B.default.weight": b.clone(),
        "classifier.weight": classifier.clone(),
    }
    product_before = state["encoder.layer.0.attn.lora_B.default.weight"] @ state[
        "encoder.layer.0.attn.lora_A.default.weight"
    ]

    stats = gauge.apply_orthogonal_a_gauge(state)

    a_after = state["encoder.layer.0.attn.lora_A.default.weight"]
    b_after = state["encoder.layer.0.attn.lora_B.default.weight"]
    assert torch.allclose(b_after @ a_after, product_before, atol=1e-5, rtol=1e-5)
    assert torch.allclose(a_after @ a_after.T, torch.eye(4), atol=1e-5, rtol=1e-5)
    assert torch.equal(state["classifier.weight"], classifier)
    assert stats["gauge_pair_count"] == 1


def test_gauge_from_env_is_disabled_by_default() -> None:
    gauge = load_gauge_module()
    state = {
        "block.lora_A.default.weight": torch.randn(2, 5),
        "block.lora_B.default.weight": torch.randn(3, 2),
    }
    before = {key: value.clone() for key, value in state.items()}

    stats = gauge.apply_lora_gauge_from_env(state, environ={})

    assert stats == {}
    for key, value in state.items():
        assert torch.equal(value, before[key])


def test_clients_avg_aggregator_applies_env_gauge(monkeypatch) -> None:
    sys.path.insert(0, str(ROOT / "code/harness/rolora-supplement/RoLoRA-code"))
    from federatedscope.core.aggregators.clients_avg_aggregator import (  # noqa: E402
        ClientsAvgAggregator,
    )

    cfg = SimpleNamespace(
        federate=SimpleNamespace(ignore_weight=False, use_ss=False),
    )
    aggregator = ClientsAvgAggregator(config=cfg)
    a1 = torch.tensor([[2.0, 0.5, -1.0], [0.25, 1.0, 0.0]])
    b1 = torch.tensor([[1.0, 0.2], [-0.5, 0.75]])
    a2 = torch.tensor([[0.5, -1.0, 1.5], [1.0, 0.25, -0.25]])
    b2 = torch.tensor([[0.25, 1.0], [1.5, -0.25]])
    avg_a = (a1 + 3 * a2) / 4
    avg_b = (b1 + 3 * b2) / 4
    product_before_gauge = avg_b @ avg_a

    monkeypatch.setenv("SLS_LORA_GAUGE", "orthogonal_a")
    result = aggregator.aggregate(
        {
            "round": 0,
            "client_feedback": [
                (
                    1,
                    {
                        "block.lora_A.default.weight": a1.clone(),
                        "block.lora_B.default.weight": b1.clone(),
                    },
                ),
                (
                    3,
                    {
                        "block.lora_A.default.weight": a2.clone(),
                        "block.lora_B.default.weight": b2.clone(),
                    },
                ),
            ],
        }
    )

    a_after = result["block.lora_A.default.weight"]
    b_after = result["block.lora_B.default.weight"]
    assert torch.allclose(b_after @ a_after, product_before_gauge, atol=1e-5, rtol=1e-5)
    assert torch.allclose(a_after @ a_after.T, torch.eye(2), atol=1e-5, rtol=1e-5)


class DummyServerModel:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


class DummyAdapterLikeServerModel:
    def __init__(self, full_state, trainable_state):
        self._full_state = full_state
        self._trainable_state = trainable_state

    def state_dict(self, return_trainable=True):
        return self._trainable_state if return_trainable else self._full_state


def test_clients_avg_gauge_uses_server_state_for_partial_rolora_payload(monkeypatch) -> None:
    sys.path.insert(0, str(ROOT / "code/harness/rolora-supplement/RoLoRA-code"))
    from federatedscope.core.aggregators.clients_avg_aggregator import (  # noqa: E402
        ClientsAvgAggregator,
    )

    cfg = SimpleNamespace(
        federate=SimpleNamespace(ignore_weight=False, use_ss=False),
    )
    server_a = torch.tensor([[2.0, 0.5, -1.0], [0.25, 1.0, 0.0]])
    server_b = torch.tensor([[1.0, 0.2], [-0.5, 0.75]])
    client_b1 = torch.tensor([[0.25, 1.0], [1.5, -0.25]])
    client_b2 = torch.tensor([[1.25, -0.5], [0.5, 0.25]])
    avg_b = (client_b1 + client_b2) / 2
    product_before_gauge = avg_b @ server_a
    model = DummyServerModel(
        {
            "block.lora_A.default.weight": server_a.clone(),
            "block.lora_B.default.weight": server_b.clone(),
        }
    )
    aggregator = ClientsAvgAggregator(model=model, config=cfg)

    monkeypatch.setenv("SLS_LORA_GAUGE", "orthogonal_a")
    result = aggregator.aggregate(
        {
            "round": 0,
            "client_feedback": [
                (1, {"block.lora_B.default.weight": client_b1.clone()}),
                (1, {"block.lora_B.default.weight": client_b2.clone()}),
            ],
        }
    )

    assert set(result) == {
        "block.lora_A.default.weight",
        "block.lora_B.default.weight",
    }
    a_after = result["block.lora_A.default.weight"]
    b_after = result["block.lora_B.default.weight"]
    assert torch.allclose(b_after @ a_after, product_before_gauge, atol=1e-5, rtol=1e-5)
    assert torch.allclose(a_after @ a_after.T, torch.eye(2), atol=1e-5, rtol=1e-5)


def test_clients_avg_gauge_requests_full_adapter_state_when_available(monkeypatch) -> None:
    sys.path.insert(0, str(ROOT / "code/harness/rolora-supplement/RoLoRA-code"))
    from federatedscope.core.aggregators.clients_avg_aggregator import (  # noqa: E402
        ClientsAvgAggregator,
    )

    cfg = SimpleNamespace(
        federate=SimpleNamespace(ignore_weight=False, use_ss=False),
    )
    server_a = torch.tensor([[2.0, 0.5, -1.0], [0.25, 1.0, 0.0]])
    server_b = torch.tensor([[1.0, 0.2], [-0.5, 0.75]])
    client_b = torch.tensor([[0.25, 1.0], [1.5, -0.25]])
    model = DummyAdapterLikeServerModel(
        full_state={
            "block.lora_A.default.weight": server_a.clone(),
            "block.lora_B.default.weight": server_b.clone(),
        },
        trainable_state={
            "block.lora_B.default.weight": server_b.clone(),
        },
    )
    aggregator = ClientsAvgAggregator(model=model, config=cfg)

    monkeypatch.setenv("SLS_LORA_GAUGE", "orthogonal_a")
    result = aggregator.aggregate(
        {
            "round": 0,
            "client_feedback": [
                (1, {"block.lora_B.default.weight": client_b.clone()}),
            ],
        }
    )

    assert "block.lora_A.default.weight" in result
    assert torch.allclose(
        result["block.lora_B.default.weight"] @ result["block.lora_A.default.weight"],
        client_b @ server_a,
        atol=1e-5,
        rtol=1e-5,
    )
