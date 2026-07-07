"""Vector math helpers (L2 normalization) used by embedding/retrieval code."""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors along rows.

    - If x is 1D, treats it as a single row and returns shape (1, dim).
    - If x is 2D, normalizes each row in-place-like and returns shape (n, dim).
    - Returns float32.
    """
    if x.size == 0:
        return x
    if x.ndim == 1:
        x = x[None, :]
    x = x.astype("float32", copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (x / norms).astype("float32", copy=False)


