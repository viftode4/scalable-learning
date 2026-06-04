from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ADAPTER_BUILDER = (
    Path(__file__).resolve().parents[1]
    / "code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/model/adapter_builder.py"
)


def load_adapter_builder():
    spec = importlib.util.spec_from_file_location("sls_adapter_builder", ADAPTER_BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyPeftModel:
    def __init__(self) -> None:
        self.params = {
            "encoder.layer.0.lora_A.default.weight": torch.nn.Parameter(torch.randn(4, 8)),
            "encoder.layer.0.lora_B.default.weight": torch.nn.Parameter(torch.randn(8, 4)),
            "classifier.weight": torch.nn.Parameter(torch.randn(2, 8)),
        }

    def named_parameters(self):
        return self.params.items()


def test_orthogonal_a_initialises_a_and_preserves_zero_b_convention() -> None:
    adapter_builder = load_adapter_builder()
    model = DummyPeftModel()
    classifier_before = model.params["classifier.weight"].detach().clone()

    returned = adapter_builder._apply_sls_lora_init(model, "orthogonal_a")

    assert returned is model
    a_weight = model.params["encoder.layer.0.lora_A.default.weight"].detach()
    b_weight = model.params["encoder.layer.0.lora_B.default.weight"].detach()
    gram = a_weight @ a_weight.T
    assert torch.allclose(gram, torch.eye(4), atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(b_weight).item() == 0
    assert torch.equal(model.params["classifier.weight"].detach(), classifier_before)


def test_default_sls_lora_init_is_noop() -> None:
    adapter_builder = load_adapter_builder()
    model = DummyPeftModel()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    returned = adapter_builder._apply_sls_lora_init(model, "default")

    assert returned is model
    for name, param in model.named_parameters():
        assert torch.equal(param.detach(), before[name])
