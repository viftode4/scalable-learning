from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

HELPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "code/harness/rolora-supplement/RoLoRA-code/"
    / "federatedscope/llm/trainer/sls_lora_lr.py"
)
spec = importlib.util.spec_from_file_location("sls_lora_lr", HELPER_PATH)
assert spec is not None and spec.loader is not None
sls_lora_lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sls_lora_lr)


class DummyLoRAModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.ones(2, 3))
        self.lora_B = torch.nn.Parameter(torch.ones(3, 2))
        self.classifier = torch.nn.Parameter(torch.ones(2, 2))

    def named_parameters(self, *args, **kwargs):  # noqa: D102
        yield "encoder.lora_A.default.weight", self.lora_A
        yield "encoder.lora_B.default.weight", self.lora_B
        yield "classifier.weight", self.classifier


def test_lora_lr_default_returns_model(monkeypatch) -> None:
    monkeypatch.delenv("SLS_LORA_LR_A", raising=False)
    monkeypatch.delenv("SLS_LORA_LR_B", raising=False)
    model = DummyLoRAModule()

    target, summary = sls_lora_lr.lora_lr_optimizer_target(model, 0.01)

    assert target is model
    assert summary is None


def test_lora_lr_env_builds_only_trainable_param_groups(monkeypatch) -> None:
    monkeypatch.setenv("SLS_LORA_LR_A", "0.001")
    monkeypatch.setenv("SLS_LORA_LR_B", "0.02")
    model = DummyLoRAModule()
    model.lora_A.requires_grad = False

    target, summary = sls_lora_lr.lora_lr_optimizer_target(model, 0.01)

    assert isinstance(target, list)
    assert summary == {
        "lora_B": {"lr": 0.02, "params": 1},
        "base": {"lr": 0.01, "params": 1},
    }
    lrs = sorted(group["lr"] for group in target)
    assert lrs == [0.01, 0.02]
    assert all(
        all(param is not model.lora_A for param in group["params"])
        for group in target
    )
