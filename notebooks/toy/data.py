"""Federated data splits for the toy MNIST experiments.

`label_split` is paper §4.2-faithful at (5c×2-label) and (10c×1-label) — those
two settings are bit-identical to the pre-refactor implementation under a fixed
rng seed (regression-tested). It additionally supports repetition
((num_clients × labels_per_client) % num_classes == 0): when more than one
client must own the same class, that class's samples are sharded across its
owners. This is how we add the (20c×1) and (50c×1) stress points without
introducing a Dirichlet splitter.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Subset


def iid_split(dataset, num_clients: int, rng: np.random.Generator) -> list[Subset]:
    n = len(dataset)
    idx = rng.permutation(n)
    chunks = np.array_split(idx, num_clients)
    return [Subset(dataset, c.tolist()) for c in chunks]


def label_split(
    dataset,
    num_clients: int,
    labels_per_client: int,
    rng: np.random.Generator,
    num_classes: int = 10,
) -> list[Subset]:
    """Federated label-non-IID split. Two regimes:

    1. ``owners_per_class == 1`` (the paper settings, e.g. 5c×2, 10c×1): each
       class goes to exactly one client. Bit-identical to the original
       implementation under a fixed rng seed.
    2. ``owners_per_class >= 2`` (e.g. 20c×1 → 2 owners/class; 50c×1 → 5
       owners/class): each class's samples are partitioned evenly across its
       owning clients.

    Raises if ``(num_clients * labels_per_client) % num_classes != 0``.
    """
    total_slots = num_clients * labels_per_client
    if total_slots % num_classes != 0:
        raise ValueError(
            "label_split requires (num_clients * labels_per_client) to be a "
            f"multiple of num_classes (={num_classes}); got "
            f"{num_clients} * {labels_per_client} = {total_slots}"
        )
    owners_per_class = total_slots // num_classes
    targets = np.array([int(dataset[i][1]) for i in range(len(dataset))])

    if owners_per_class == 1:
        # Paper-faithful path. Preserved verbatim so the existing seed-stable
        # 5c×2 / 10c×1 reproductions match byte-for-byte.
        order = rng.permutation(num_classes)
        subsets = []
        for c in range(num_clients):
            own = order[c * labels_per_client : (c + 1) * labels_per_client]
            idx = np.where(np.isin(targets, own))[0]
            subsets.append(Subset(dataset, idx.tolist()))
        return subsets

    # Repetition path: build a length-(total_slots) sequence where each class
    # appears `owners_per_class` times, permute it once, then reshape into a
    # (num_clients, labels_per_client) assignment matrix.
    base = np.repeat(np.arange(num_classes), owners_per_class)
    order = rng.permutation(base)
    assignments = order.reshape(num_clients, labels_per_client)

    class_owners: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    for client_idx, labels in enumerate(assignments):
        for label in labels:
            class_owners[int(label)].append(client_idx)

    client_indices: dict[int, list[int]] = {c: [] for c in range(num_clients)}
    for c in range(num_classes):
        class_idx = np.where(targets == c)[0]
        # Deterministic shard order — no extra rng draws — keeps the rng state
        # consumed by this function strictly equal to one permutation.
        shards = np.array_split(class_idx, owners_per_class)
        for owner, shard in zip(class_owners[c], shards, strict=True):
            client_indices[owner].extend(shard.tolist())

    return [Subset(dataset, sorted(client_indices[c])) for c in range(num_clients)]
