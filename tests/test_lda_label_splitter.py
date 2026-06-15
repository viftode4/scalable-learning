"""Regression test for the lda_label splitter (the 5h-hang fix).

The generic `lda` splitter skews over `x['categories']`, which for the LLM
datasets is the per-paragraph code (~unique per example, ~105k "classes" on
QNLI). That makes dirichlet slicing hang for hours. `lda_label` skews over the
scalar task label instead. This test pins: (1) it terminates fast even when
`categories` is unique per item, (2) it produces real label skew at small
alpha, and (3) larger alpha is less skewed.

Runs only where federatedscope + torch import (the supplement venv); skipped
otherwise so the lightweight `uv run pytest` suite stays green.
"""

from __future__ import annotations

import signal
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "code/harness/rolora-supplement/RoLoRA-code"


@pytest.fixture(scope="module")
def splitter_mod():
    import sys
    sys.path.insert(0, str(SUPP))
    torch = pytest.importorskip("torch")
    try:
        from federatedscope.contrib.splitter import lda_label_splitter as m
    except Exception as err:  # federatedscope not importable in this env
        pytest.skip(f"federatedscope unavailable: {err}")
    return m, torch


def test_scalar_label_coercions(splitter_mod):
    m, torch = splitter_mod
    assert m._scalar_label({"labels": 1, "categories": 99999}) == 1
    assert m._scalar_label({"categories": 3}) == 3
    assert m._scalar_label({"labels": torch.tensor(0)}) == 0
    assert m._scalar_label(("x", 1)) == 1


def test_does_not_hang_and_skews(splitter_mod):
    m, _ = splitter_mod

    def _timeout(*_a):
        raise TimeoutError("lda_label splitter did not terminate")

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(20)
    try:
        N = 4000
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, size=N)
        # categories UNIQUE per item -> the exact shape that hangs `lda`.
        data = [{"labels": int(y[i]), "categories": i} for i in range(N)]

        shards = m.LDALabelSplitter(client_num=6, alpha=0.1)(data)
        sizes = [len(s) for s in shards]
        assert len(shards) == 6
        assert sum(sizes) == N
        assert all(sz > 0 for sz in sizes), sizes

        fracs = [float(np.mean([d["labels"] for d in s])) for s in shards]
        spread = max(fracs) - min(fracs)
        assert spread > 0.15, f"expected label skew, got spread={spread}"

        fracs_iid = [
            float(np.mean([d["labels"] for d in s]))
            for s in m.LDALabelSplitter(client_num=6, alpha=100.0)(data)
        ]
        assert (max(fracs_iid) - min(fracs_iid)) < spread
    finally:
        signal.alarm(0)
