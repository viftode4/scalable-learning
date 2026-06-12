from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SLS_MONITOR_PATH = (
    ROOT
    / "code/harness/rolora-supplement/RoLoRA-code/federatedscope/core/sls_monitor.py"
)
ROUND_LOGGING_PATHS = [
    ROOT
    / "code/harness/rolora-supplement/RoLoRA-code/federatedscope/core/sls_monitor.py",
    ROOT
    / "code/harness/rolora-supplement/RoLoRA-code/federatedscope/core/workers/client.py",
    ROOT
    / "code/harness/rolora-supplement/RoLoRA-code/federatedscope/core/workers/server.py",
]
TRAINER_PATH = (
    ROOT
    / "code/harness/rolora-supplement/RoLoRA-code/federatedscope/llm/trainer/trainer.py"
)


def load_sls_monitor():
    spec = importlib.util.spec_from_file_location("sls_monitor_for_wandb_test", SLS_MONITOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def install_fake_wandb(monkeypatch):
    fake = ModuleType("wandb")
    fake.run = object()
    fake.defined_metrics = []
    fake.logged = []

    def define_metric(*args, **kwargs):
        fake.defined_metrics.append((args, kwargs))

    def log(payload, **kwargs):
        fake.logged.append((payload, kwargs))

    fake.define_metric = define_metric
    fake.log = log
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_round_logger_uses_custom_round_axis_without_global_step(monkeypatch):
    monitor = load_sls_monitor()
    fake_wandb = install_fake_wandb(monkeypatch)

    monitor.wandb_log_round(
        {"train_acc": 0.75, "train_loss": 1.25, "round": 7, "client": 1, "note": "skip"},
        round_num=7,
        namespace="client_01",
    )

    assert fake_wandb.defined_metrics == [
        (("client_01/round",), {}),
        (("client_01/*",), {"step_metric": "client_01/round"}),
    ]
    assert fake_wandb.logged == [
        (
            {
                "client_01/train_acc": 0.75,
                "client_01/train_loss": 1.25,
                "client_01/client": 1,
                "client_01/round": 7,
            },
            {},
        )
    ]


def test_monitor_wandb_log_still_respects_sls_monitor_gate(monkeypatch):
    monitor = load_sls_monitor()
    fake_wandb = install_fake_wandb(monkeypatch)
    monkeypatch.delenv("SLS_MONITOR", raising=False)

    monitor.wandb_log({"value": 1.0, "round": 3}, step=3, namespace="monitor/example")

    assert fake_wandb.logged == []

    monkeypatch.setenv("SLS_MONITOR", "1")
    monitor.wandb_log({"value": 1.0, "round": 3}, step=3, namespace="monitor/example")

    assert fake_wandb.logged == [
        ({"monitor/example/value": 1.0, "monitor/example/round": 3}, {})
    ]


def test_round_metrics_do_not_use_wandb_global_step() -> None:
    for path in ROUND_LOGGING_PATHS:
        source = path.read_text()
        assert not re.search(r"(?<![A-Za-z0-9_])wandb\.log\([^\n]*step\s*=", source), path


def test_experiment_configs_do_not_enable_upstream_wandb_logger() -> None:
    for path in (ROOT / "experiments/configs").glob("*.yaml"):
        source = path.read_text()
        assert not re.search(r"(?im)^wandb:\s*(?:\n\s+.*)*?\n\s+use:\s*true\b", source), path
        assert not re.search(r"(?im)^wandb\.\s*use\s*[:=]\s*true\b", source), path


def test_custom_trainer_uses_explicit_wandb_entity_env() -> None:
    source = TRAINER_PATH.read_text()

    assert "entity=os.environ.get('WANDB_ENTITY')" in source
