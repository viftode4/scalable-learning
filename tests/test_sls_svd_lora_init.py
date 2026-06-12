from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

ADAPTER_BUILDER = (
    Path(__file__).resolve().parents[1]
    / "code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/model/adapter_builder.py"
)


def load_adapter_builder():
    spec = importlib.util.spec_from_file_location("sls_adapter_builder_svd", ADAPTER_BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyLoraLinear(nn.Module):
    def __init__(self, *, in_features: int = 5, out_features: int = 6, rank: int = 2, scaling: float = 4.0) -> None:
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, out_features, bias=False)})
        self.active_adapters = ["default"]
        self.r = {"default": rank}
        self.scaling = {"default": scaling}
        self.fan_in_fan_out = False
        with torch.no_grad():
            values = torch.linspace(-1.25, 1.75, steps=out_features * in_features)
            self.base_layer.weight.copy_(values.reshape(out_features, in_features))
            self.lora_A["default"].weight.zero_()
            self.lora_B["default"].weight.zero_()

    def get_base_layer(self):
        return self.base_layer

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        return self.scaling[adapter] * (
            self.lora_B[adapter].weight @ self.lora_A[adapter].weight
        )


class DummyPeftModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora = DummyLoraLinear()
        self.classifier = nn.Linear(6, 2, bias=False)


def top_rank_reconstruction(weight: torch.Tensor, rank: int) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return (u[:, :rank] * s[:rank].unsqueeze(0)) @ vh[:rank, :]


def test_svd_compensated_initialises_nonzero_factors_and_preserves_effective_weight() -> None:
    adapter_builder = load_adapter_builder()
    model = DummyPeftModel()
    layer = model.lora
    original = layer.base_layer.weight.detach().clone()

    returned = adapter_builder._apply_sls_lora_init(model, "svd_compensated")

    assert returned is model
    assert layer.lora_A["default"].weight.detach().abs().max().item() > 0.0
    assert layer.lora_B["default"].weight.detach().abs().max().item() > 0.0

    delta_after = layer.get_delta_weight("default").detach()
    effective_after = layer.base_layer.weight.detach() + delta_after
    expected_top_rank = top_rank_reconstruction(original, rank=2)

    assert torch.allclose(delta_after, expected_top_rank, atol=1e-5, rtol=1e-5)
    assert torch.allclose(effective_after, original, atol=1e-5, rtol=1e-5)


def test_svd_compensated_aliases_normalise_to_same_variant() -> None:
    adapter_builder = load_adapter_builder()

    for value in ("svd", "svd-compensated", "pissa", "pissa_compensated"):
        assert adapter_builder._normalise_sls_lora_init(value) == "svd_compensated"


def test_svd_compensated_rejects_fan_in_fan_out_layers() -> None:
    adapter_builder = load_adapter_builder()
    model = DummyPeftModel()
    model.lora.fan_in_fan_out = True

    with pytest.raises(ValueError, match="fan_in_fan_out"):
        adapter_builder._apply_sls_lora_init(model, "svd_compensated")


def test_svd_compensated_rejects_second_application_to_same_model() -> None:
    adapter_builder = load_adapter_builder()
    model = DummyPeftModel()

    adapter_builder._apply_sls_lora_init(model, "svd_compensated")

    with pytest.raises(RuntimeError, match="already applied"):
        adapter_builder._apply_sls_lora_init(model, "svd_compensated")


class LegacyDummyLoraLinear(nn.Module):
    """PEFT 0.3-style LoRA layer: the frozen base weight lives on module.weight."""

    def __init__(self, *, in_features: int = 5, out_features: int = 6, rank: int = 2, scaling: float = 4.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, out_features, bias=False)})
        self.active_adapter = "default"
        self.r = {"default": rank}
        self.scaling = {"default": scaling}
        self.fan_in_fan_out = False
        with torch.no_grad():
            values = torch.linspace(-0.75, 2.25, steps=out_features * in_features)
            self.weight.copy_(values.reshape(out_features, in_features))
            self.lora_A["default"].weight.zero_()
            self.lora_B["default"].weight.zero_()


class LegacyDummyPeftModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora = LegacyDummyLoraLinear()


def test_svd_compensated_supports_peft_0_3_style_layers_with_inline_weight() -> None:
    adapter_builder = load_adapter_builder()
    model = LegacyDummyPeftModel()
    layer = model.lora
    original = layer.weight.detach().clone()

    adapter_builder._apply_sls_lora_init(model, "svd_compensated")

    delta_after = layer.scaling["default"] * (
        layer.lora_B["default"].weight @ layer.lora_A["default"].weight
    )
    effective_after = layer.weight.detach() + delta_after.detach()
    expected_top_rank = top_rank_reconstruction(original, rank=2)

    assert torch.allclose(delta_after, expected_top_rank, atol=1e-5, rtol=1e-5)
    assert torch.allclose(effective_after, original, atol=1e-5, rtol=1e-5)
