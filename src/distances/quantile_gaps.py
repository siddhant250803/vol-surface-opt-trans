"""Quantile gap metrics."""

from __future__ import annotations

import numpy as np


def _quantile_from_density(k: np.ndarray, q: np.ndarray, u: float) -> float:
    """Inverse CDF at level u."""
    if len(k) < 2:
        return float(k[0]) if len(k) == 1 else np.nan
    dx = np.diff(k)
    cdf = np.concatenate([[0], np.cumsum(q[:-1] * dx)])
    total = cdf[-1] + q[-1] * dx[-1] if len(dx) > 0 else cdf[-1]
    if total <= 0:
        return float(k[0])
    cdf = cdf / total
    return float(np.interp(u, cdf, k))


def quantile_gap(k: np.ndarray, q1: np.ndarray, q2: np.ndarray, level: float) -> float:
    """|Q1^{-1}(level) - Q2^{-1}(level)|."""
    q1_inv = _quantile_from_density(k, q1, level)
    q2_inv = _quantile_from_density(k, q2, level)
    return float(np.abs(q1_inv - q2_inv))
