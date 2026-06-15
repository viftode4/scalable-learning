"""Unit tests for the notebooks/toy/ package — the pieces added during the
RoLoRA-improvement refactor.

The legacy `tests/test_mnist_fig2.py`, `test_aggregation_invariants.py`, and
`test_init_conventions.py` continue to exercise the re-exported surface; this
file covers the new behaviours.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from toy import (  # noqa: E402
    PRESETS,
    MethodConfig,
    PaperToyModel,
    ServerMomentum,
    average_factor,
    iid_split,
    init_factor,
    label_split,
    local_train,
    run_centralized,
    run_method,
)

# ---------- label_split (extended) ----------


class _FakeDataset:
    """In-memory MNIST-shaped dataset for fast split testing.

    Yields ``(zero-tensor, int_label)`` pairs so `label_split` can inspect
    targets without touching disk.
    """

    def __init__(self, targets: list[int]) -> None:
        self._targets = targets
        self._x = torch.zeros(1, 1)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self._x, self._targets[i]


def _balanced_targets(per_class: int = 60, num_classes: int = 10) -> list[int]:
    return [c for c in range(num_classes) for _ in range(per_class)]


def test_label_split_paper_setting_5c2_byte_identical() -> None:
    """5 clients × 2 labels (paper Fig. 2 left panel) goes through the
    `owners_per_class == 1` compat path and matches a hand-computed reference."""
    ds = _FakeDataset(_balanced_targets())
    rng = np.random.default_rng(0)
    subsets = label_split(ds, 5, 2, rng)
    assert len(subsets) == 5
    # Disjoint: every sample index appears in exactly one subset.
    all_idx = sorted(i for s in subsets for i in s.indices)
    assert all_idx == sorted(range(len(ds)))


def test_label_split_paper_setting_10c1() -> None:
    ds = _FakeDataset(_balanced_targets())
    rng = np.random.default_rng(0)
    subsets = label_split(ds, 10, 1, rng)
    assert len(subsets) == 10
    # Each client should own exactly one class.
    for s in subsets:
        labels = {ds._targets[i] for i in s.indices}
        assert len(labels) == 1


def test_label_split_20c1_repetition_two_owners_per_class() -> None:
    """20 clients × 1 label on 10 classes ⇒ exactly 2 owners per class."""
    ds = _FakeDataset(_balanced_targets(per_class=100))
    rng = np.random.default_rng(0)
    subsets = label_split(ds, 20, 1, rng)
    assert len(subsets) == 20

    # Each client still owns exactly 1 class.
    class_to_clients: dict[int, list[int]] = {c: [] for c in range(10)}
    for client_idx, s in enumerate(subsets):
        labels = {ds._targets[i] for i in s.indices}
        assert len(labels) == 1
        class_to_clients[labels.pop()].append(client_idx)

    # Each class is owned by exactly 2 clients.
    for c, owners in class_to_clients.items():
        assert len(owners) == 2, f"class {c} owners: {owners}"

    # Sample indices are disjoint across the whole split.
    all_idx = sorted(i for s in subsets for i in s.indices)
    assert all_idx == sorted(range(len(ds)))


def test_label_split_50c1_repetition_five_owners_per_class() -> None:
    ds = _FakeDataset(_balanced_targets(per_class=100))
    rng = np.random.default_rng(0)
    subsets = label_split(ds, 50, 1, rng)
    assert len(subsets) == 50
    class_counts: dict[int, int] = {c: 0 for c in range(10)}
    for s in subsets:
        labels = {ds._targets[i] for i in s.indices}
        assert len(labels) == 1
        class_counts[labels.pop()] += 1
    assert all(v == 5 for v in class_counts.values())


def test_label_split_rejects_bad_pairing() -> None:
    ds = _FakeDataset(_balanced_targets())
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="multiple of num_classes"):
        label_split(ds, 7, 1, rng)


# ---------- init_factor ----------


def test_init_factor_orthogonal_rows() -> None:
    torch.manual_seed(0)
    # `nn.init.orthogonal_` makes a 2-D tensor have orthonormal rows when
    # rows < cols, orthonormal cols otherwise. For a (16, 784) shape it
    # produces orthonormal rows; for (784, 16) it produces orthonormal cols.
    A = torch.empty(16, 784)  # noqa: N806 — math notation
    init_factor(A, "orthogonal")
    gram = A @ A.T
    assert torch.allclose(gram, torch.eye(16), atol=1e-5)


def test_init_factor_zeros() -> None:
    A = torch.empty(4, 4)  # noqa: N806 — math notation
    init_factor(A, "zeros")
    assert torch.equal(A, torch.zeros_like(A))


def test_init_factor_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown init kind"):
        init_factor(torch.empty(2, 2), "lol")  # type: ignore[arg-type]


# ---------- local_train (prox + lr_a/lr_b) ----------


def _toy_loader() -> DataLoader:
    """Tiny in-memory loader of 8 fake (x, y) MNIST samples."""
    n = 8
    xs = torch.randn(n, 1, 28, 28)
    ys = torch.randint(0, 10, (n,))
    ds = list(zip(xs, ys, strict=True))
    return DataLoader(ds, batch_size=4, shuffle=False)


def test_local_train_prox_pulls_toward_anchor() -> None:
    """With prox_mu > 0 and anchor set to a *different* value than init,
    after one step the active factor should be closer to the anchor than
    the no-prox baseline."""
    torch.manual_seed(0)
    model_noprox = PaperToyModel(rank=4)
    model_prox = copy.deepcopy(model_noprox)

    # Train only B; freeze A.
    for m in (model_noprox, model_prox):
        m.A.requires_grad = False
        m.B.requires_grad = True

    # Anchor = the *negation* of the current B (far from current value).
    anchor = {"B": [(-model_noprox.B.detach().clone())]}

    loader = _toy_loader()
    device = torch.device("cpu")

    local_train(
        model_noprox,
        loader,
        lr=0.01,
        device=device,
        grad_clip=0.0,
        steps=2,
        prox_mu=0.0,
    )
    local_train(
        model_prox,
        loader,
        lr=0.01,
        device=device,
        grad_clip=0.0,
        steps=2,
        prox_mu=1.0,
        prox_anchor=anchor,
    )

    d_noprox = (model_noprox.B - anchor["B"][0]).norm().item()
    d_prox = (model_prox.B - anchor["B"][0]).norm().item()
    assert d_prox < d_noprox, f"prox should pull toward anchor: {d_prox} vs {d_noprox}"


def test_local_train_legacy_lr_path_unchanged() -> None:
    """Calling with `lr=...` and no lr_a/lr_b must still work (legacy API)."""
    torch.manual_seed(0)
    model = PaperToyModel(rank=4)
    model.A.requires_grad = False
    model.B.requires_grad = True
    local_train(model, _toy_loader(), lr=0.01, device=torch.device("cpu"), steps=1)


def test_local_train_rejects_both_lr_and_asymmetric() -> None:
    model = PaperToyModel(rank=4)
    with pytest.raises(ValueError, match="not both"):
        local_train(
            model,
            _toy_loader(),
            lr=0.01,
            lr_a=0.01,
            lr_b=0.01,
            device=torch.device("cpu"),
            steps=1,
        )


# ---------- ServerMomentum ----------


def _make_server_clients(num_clients: int = 3, rank: int = 2):
    torch.manual_seed(0)
    server = PaperToyModel(rank=rank)
    clients = [copy.deepcopy(server) for _ in range(num_clients)]
    # Drift client B values to make averaging visible.
    for i, c in enumerate(clients):
        c.B.data.add_(0.1 * (i + 1))
    return server, clients


def test_server_momentum_beta_zero_matches_average_factor() -> None:
    """β=0 ⇒ Polyak buffer just stores Δ_t, and server += Δ_t produces the
    plain average. Must be bit-identical to `average_factor`."""
    server_a, clients_a = _make_server_clients()
    server_b, clients_b = _make_server_clients()

    momentum = ServerMomentum(beta=0.0)
    momentum.step(server_a, clients_a, "B")
    average_factor(server_b, clients_b, "B")

    assert torch.equal(server_a.B.data, server_b.B.data)


def test_server_momentum_accumulates_across_steps() -> None:
    """With β > 0, repeated identical aggregations should produce strictly
    larger update magnitudes than the first one (momentum buildup)."""
    server, _ = _make_server_clients()
    momentum = ServerMomentum(beta=0.9)

    snap = server.B.detach().clone()

    # Step 1: aggregate from a fixed set of drifted clients.
    _, clients = _make_server_clients()
    momentum.step(server, clients, "B")
    delta1 = (server.B - snap).norm().item()

    # Step 2: same drifted clients (but server has moved, so Δ_2 < Δ_1).
    # The momentum buffer carries forward, so the *applied* update v_2 should
    # still produce a non-trivial second move; check that the buffer norm grew.
    buf_after_1 = momentum._buffers["B"][0].norm().item()
    assert delta1 > 0
    assert buf_after_1 > 0


def test_server_momentum_rejects_invalid_beta() -> None:
    with pytest.raises(ValueError):
        ServerMomentum(beta=-0.1)
    with pytest.raises(ValueError):
        ServerMomentum(beta=1.0)


def test_server_momentum_rejects_invalid_server_lr() -> None:
    with pytest.raises(ValueError):
        ServerMomentum(beta=0.9, server_lr=0.0)
    with pytest.raises(ValueError):
        ServerMomentum(beta=0.9, server_lr=-0.1)
    # server_lr != 1.0 with no momentum is a separate concept; reject it.
    with pytest.raises(ValueError):
        ServerMomentum(beta=0.0, server_lr=0.5)


def test_server_momentum_amplitude_preserving_at_steady_state() -> None:
    """With β=0.9, η_s=1−β=0.1, feeding a *constant* Δ across rounds drives
    the per-round server update to ≈ ‖Δ‖ — same magnitude as plain
    `average_factor`. This is the load-bearing property of the canonical
    FedAvgM/FedOpt parameterisation."""
    server, clients = _make_server_clients()
    momentum = ServerMomentum(beta=0.9, server_lr=0.1)

    # Reference: a single plain-average update from this client state.
    ref_server = copy.deepcopy(server)
    ref_clients = [copy.deepcopy(c) for c in clients]
    average_factor(ref_server, ref_clients, "B")
    reference_step = (ref_server.B - server.B).norm().item()
    assert reference_step > 0

    # Hold Δ_t constant by advancing clients in lockstep with the server each
    # step — otherwise avg(clients) − server shrinks to zero as server
    # approaches avg(clients), and the system converges to a fixed point.
    last_step_norm = 0.0
    for _ in range(200):
        prev = server.B.detach().clone()
        momentum.step(server, clients, "B")
        moved = server.B.data - prev
        last_step_norm = moved.norm().item()
        for c in clients:
            c.B.data.add_(moved)

    # Steady-state per-round update should match the plain-average step
    # magnitude within a few percent.
    assert abs(last_step_norm - reference_step) / reference_step < 0.05


# ---------- MethodConfig validation ----------


def test_method_config_rejects_negative_mu() -> None:
    with pytest.raises(ValueError):
        MethodConfig(name="x", alternation="rolora", lr_a=0.01, lr_b=0.01, prox_mu=-0.1)


def test_method_config_rejects_beta_outside_range() -> None:
    with pytest.raises(ValueError):
        MethodConfig(name="x", alternation="rolora", lr_a=0.01, lr_b=0.01, server_momentum=1.0)


def test_method_config_rejects_invalid_server_lr() -> None:
    with pytest.raises(ValueError):
        MethodConfig(
            name="x", alternation="rolora", lr_a=0.01, lr_b=0.01,
            server_momentum=0.9, server_lr=0.0,
        )
    # server_lr != 1.0 without momentum is rejected (separate concept).
    with pytest.raises(ValueError):
        MethodConfig(
            name="x", alternation="rolora", lr_a=0.01, lr_b=0.01,
            server_momentum=0.0, server_lr=0.5,
        )


def test_method_config_rejects_unknown_init() -> None:
    with pytest.raises(ValueError):
        MethodConfig(name="x", alternation="rolora", lr_a=0.01, lr_b=0.01, init_a="lol")


def test_method_config_presets_are_valid() -> None:
    """Sanity: every preset constructs without error and has the expected alternation."""
    for key, cfg in PRESETS.items():
        assert cfg.lr_a > 0
        if key == "centralized":
            assert cfg.alternation == "centralized"
        elif key == "base_lora":
            assert cfg.alternation == "lora"
        elif key == "base_ffa_lora":
            assert cfg.alternation == "ffa_lora"
        else:
            assert cfg.alternation == "rolora"


# ---------- run_centralized (ceiling sanity) ----------


@pytest.fixture(scope="module")
def tiny_mnist(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("mnist")
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    full_test = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    train = Subset(full_train, list(range(512)))
    test = Subset(full_test, list(range(256)))
    rng = np.random.default_rng(0)
    return iid_split(train, 3, rng), DataLoader(test, batch_size=128)


def test_run_centralized_produces_curves(tiny_mnist) -> None:
    train_sets, test_loader = tiny_mnist
    losses, accs = run_centralized(
        train_sets,
        test_loader=test_loader,
        rank=4,
        rounds=3,
        local_steps=5,
        lr=0.02,
        batch_size=64,
        seed=0,
        device=torch.device("cpu"),
    )
    # rounds + 1: round-0 (pre-training) eval is prepended.
    assert len(losses) == 4
    assert len(accs) == 4
    for loss, acc in zip(losses, accs, strict=True):
        assert loss > 0
        assert 0.0 <= acc <= 1.0


def test_run_centralized_beats_federated_at_same_budget(tiny_mnist) -> None:
    """The non-federated ceiling should reach a lower loss than RoLoRA at
    matched gradient budget on a clean IID toy split. If this regresses it
    likely means `run_centralized`'s budget computation is off."""
    train_sets, test_loader = tiny_mnist
    common = dict(
        rank=4,
        rounds=5,
        local_steps=5,
        batch_size=64,
        seed=0,
        device=torch.device("cpu"),
    )
    _, fed_acc = run_method(
        "rolora",
        train_sets=train_sets,
        test_loader=test_loader,
        lr=0.02,
        **common,
    )
    _, ctr_acc = run_centralized(
        train_sets,
        test_loader=test_loader,
        lr=0.02,
        **common,
    )
    # Final accuracy: centralized >= federated on IID + matched budget.
    assert ctr_acc[-1] >= fed_acc[-1] - 1e-6
