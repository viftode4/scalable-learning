"""Exactness invariants that RoLoRA depends on.

Specifically: before each round of training, the *frozen* factor (A on odd rounds, B on
even rounds) must be bit-identical across every client and the server. Otherwise the
average-of-products equality silently breaks and RoLoRA degrades to broken-LoRA.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Import the MNIST script's helpers without forcing it to be a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import mnist_fig2  # noqa: E402


def _make_server_and_clients(num_clients: int = 3, rank: int = 1):
    torch.manual_seed(0)
    server = mnist_fig2.MLP(rank)
    import copy

    clients = [copy.deepcopy(server) for _ in range(num_clients)]
    return server, clients


def test_assert_factor_identical_passes_at_init() -> None:
    server, clients = _make_server_and_clients()
    mnist_fig2.assert_factor_identical(server, clients, "A")
    mnist_fig2.assert_factor_identical(server, clients, "B")


def test_assert_factor_identical_fails_when_drifted() -> None:
    server, clients = _make_server_and_clients()
    clients[0].fc1.A.data.add_(1.0)
    with pytest.raises(AssertionError):
        mnist_fig2.assert_factor_identical(server, clients, "A")


def test_broadcast_then_invariant_holds() -> None:
    server, clients = _make_server_and_clients()
    # Drift A on every client, then broadcast from server — invariant restored.
    for i, c in enumerate(clients):
        c.fc1.A.data.add_(0.1 * (i + 1))
    mnist_fig2.broadcast(server, clients, "A")
    mnist_fig2.assert_factor_identical(server, clients, "A")


def test_average_factor_matches_manual_mean() -> None:
    server, clients = _make_server_and_clients()
    # Set client A's to known values and compare server average.
    for i, c in enumerate(clients):
        c.fc1.A.data.fill_(float(i))
    mnist_fig2.average_factor(server, clients, "A")
    expected = torch.tensor((0.0 + 1.0 + 2.0) / 3)
    assert torch.allclose(server.fc1.A.data, expected.expand_as(server.fc1.A.data))
