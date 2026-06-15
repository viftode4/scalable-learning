"""Device probe shared by every toy entry point.

Priority: CUDA → Apple MPS → CPU. Lifted out of `mnist_fig2.main()` so the
comparison and sweep scripts use the same logic without duplicating code.
"""

from __future__ import annotations

import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
