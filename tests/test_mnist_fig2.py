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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import mnist_fig2  # noqa: E402


class LabelDataset(Dataset):
    def __init__(self, labels: list[int]) -> None:
        self.targets = labels

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return torch.tensor([float(index)]), self.targets[index]


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
    assert len(losses) == 3
    assert len(accs) == 3
    for loss, acc in zip(losses, accs, strict=True):
        assert loss > 0
        assert 0.0 <= acc <= 1.0


def test_phase_pattern_repeats_bba() -> None:
    assert [mnist_fig2.phase_for_round(i, "BBA") for i in range(7)] == [
        "B",
        "B",
        "A",
        "B",
        "B",
        "A",
        "B",
    ]


def test_orthogonal_a_initialization_keeps_zero_adapter() -> None:
    torch.manual_seed(0)
    model = mnist_fig2.MLP(rank=4)
    base_outputs = [
        layer.base.weight.detach().clone()
        for layer in (model.fc1, model.fc2)
    ]
    old_fro_norms = [layer.A.detach().norm() for layer in (model.fc1, model.fc2)]

    mnist_fig2.apply_lora_initialization(model, "orthogonal_a")

    for layer, base_weight, old_fro in zip(
        (model.fc1, model.fc2), base_outputs, old_fro_norms, strict=True
    ):
        gram = layer.A.detach().T @ layer.A.detach()
        diagonal = torch.diag(gram)
        off_diagonal = gram - torch.diag(diagonal)
        assert torch.allclose(off_diagonal, torch.zeros_like(off_diagonal), atol=1e-4)
        assert torch.allclose(layer.A.detach().norm(), old_fro, rtol=1e-5)
        assert torch.equal(layer.B.detach(), torch.zeros_like(layer.B))
        assert torch.equal(layer.base.weight.detach(), base_weight)


def test_svd_compensated_initialization_preserves_effective_weight() -> None:
    torch.manual_seed(0)
    layer = mnist_fig2.LoRALinear(8, 5, rank=3)
    before = layer.effective_weight().detach().clone()

    mnist_fig2.apply_lora_initialization_to_layer(layer, "svd_compensated")

    assert layer.A.norm() > 0
    assert layer.B.norm() > 0
    assert torch.allclose(layer.effective_weight(), before, atol=1e-5)


def test_basis_transport_preserves_toy_lora_product() -> None:
    torch.manual_seed(0)
    layer = mnist_fig2.LoRALinear(12, 7, rank=3)
    mnist_fig2.apply_lora_initialization_to_layer(layer, "orthogonal_a")
    layer.B.data.normal_()
    product_before = layer.lora_product().detach().clone()
    old_a = layer.A.detach().clone()
    basis_change = torch.tensor(
        [[1.2, 0.1, -0.2], [0.0, 0.8, 0.3], [0.1, -0.2, 1.1]],
        dtype=layer.A.dtype,
    )
    new_a = old_a @ basis_change
    layer.A.data.copy_(new_a)

    stats = mnist_fig2.transport_layer_coefficients(layer, old_a)

    assert stats["transported"] is True
    assert torch.allclose(layer.lora_product(), product_before, atol=1e-5)


def test_label_shard_split_assigns_one_label_per_client() -> None:
    dataset = LabelDataset([label for label in range(10) for _ in range(6)])
    rng = np.random.default_rng(0)

    client_sets = mnist_fig2.label_shard_split(
        dataset,
        num_clients=10,
        rng=rng,
        labels_per_client=1,
    )

    assert len(client_sets) == 10
    labels_seen = set()
    for client_set in client_sets:
        labels = {dataset.targets[index] for index in client_set.indices}
        assert len(labels) == 1
        labels_seen.update(labels)
    assert labels_seen == set(range(10))
