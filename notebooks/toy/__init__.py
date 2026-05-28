"""Reusable components for the §4.2 toy federated-LoRA experiments.

`notebooks/mnist_fig2.py` is the thin CLI wrapper that consumes this package and
preserves the original Figure-2 reproduction behaviour. `mnist_fig2_compare.py`
and `toy/sweep.py` are the entry points for the RoLoRA-improvement study.
"""

from .client import evaluate, local_train
from .config import PRESETS, MethodConfig
from .data import iid_split, label_split
from .device import pick_device
from .model import (
    METHODS,
    MLP,
    LoRALinear,
    PaperToyModel,
    adapter_params,
    init_factor,
    set_factor_trainable,
)
from .rounds import run_centralized, run_method
from .server import (
    ServerMomentum,
    assert_factor_identical,
    average_factor,
    broadcast,
)

__all__ = [
    "METHODS",
    "MLP",
    "LoRALinear",
    "MethodConfig",
    "PRESETS",
    "PaperToyModel",
    "ServerMomentum",
    "adapter_params",
    "assert_factor_identical",
    "average_factor",
    "broadcast",
    "evaluate",
    "iid_split",
    "init_factor",
    "label_split",
    "local_train",
    "pick_device",
    "run_centralized",
    "run_method",
    "set_factor_trainable",
]
