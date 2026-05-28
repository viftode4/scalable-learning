"""Smoke test for notebooks/mnist_fig2.py.

Runs each method for a tiny number of rounds on a small MNIST subset and asserts
each one produces well-formed loss + accuracy curves of the expected length.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import mnist_fig2  # noqa: E402


@pytest.fixture(scope="module")
def tiny_loaders(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("mnist")
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    full_test = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    train = Subset(full_train, list(range(512)))
    test = Subset(full_test, list(range(256)))
    rng = np.random.default_rng(0)
    return mnist_fig2.iid_split(train, 3, rng), DataLoader(test, batch_size=128)


@pytest.mark.parametrize("method", mnist_fig2.METHODS)
def test_method_produces_curves(method: str, tiny_loaders) -> None:
    train_sets, test_loader = tiny_loaders
    losses, accs = mnist_fig2.run_method(
        method,
        train_sets=train_sets,
        test_loader=test_loader,
        rank=1,
        rounds=3,
        local_steps=5,
        lr=0.05,
        batch_size=64,
        seed=0,
        device=torch.device("cpu"),
    )
    # rounds + 1: round-0 (pre-training) eval is prepended to the curves.
    assert len(losses) == 4
    assert len(accs) == 4
    for loss, acc in zip(losses, accs, strict=True):
        assert loss > 0
        assert 0.0 <= acc <= 1.0
