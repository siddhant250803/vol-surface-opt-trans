"""Moment diffs, tail mass gaps."""

from __future__ import annotations

import numpy as np


def _cdf_at(k_arr: np.ndarray, q_arr: np.ndarray, x: float) -> float:
    """CDF at x: P(X <= x)."""
    if len(k_arr) < 2:
        return 0.0 if x < k_arr[0] else 1.0
    idx = np.searchsorted(k_arr, x)
    if idx == 0:
        return 0.0
    if idx >= len(k_arr):
        return 1.0
    cdf = np.trapezoid(q_arr[:idx], k_arr[:idx])
    return float(cdf)


def tail_mass_gap(
    k: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    levels: list[float],
) -> tuple[float, float]:
    """Left and right tail mass gaps. |CDF1(x) - CDF2(x)| at quantile cutoffs."""
    if len(levels) < 2:
        return np.nan, np.nan
    left_levels = [l for l in levels if l < 0.5]
    right_levels = [l for l in levels if l > 0.5]
    if not left_levels:
        left_levels = [0.01, 0.05]
    if not right_levels:
        right_levels = [0.95, 0.99]

    left_gaps = []
    for u in left_levels:
        x = np.percentile(k, u * 100)
        c1 = _cdf_at(k, q1, x)
        c2 = _cdf_at(k, q2, x)
        left_gaps.append(abs(c1 - c2))
    right_gaps = []
    for u in right_levels:
        x = np.percentile(k, u * 100)
        c1 = 1 - _cdf_at(k, q1, x)
        c2 = 1 - _cdf_at(k, q2, x)
        right_gaps.append(abs(c1 - c2))

    return float(np.mean(left_gaps)) if left_gaps else np.nan, float(np.mean(right_gaps)) if right_gaps else np.nan


def moment_diff(k: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> tuple[float, float]:
    """|E1[x] - E2[x]| and |E1[x^2] - E2[x^2]|."""
    dx = np.diff(k, prepend=k[0])
    if len(dx) != len(k):
        dx = np.diff(k)
        dx = np.concatenate([[dx[0]], dx])
    m1_1 = np.trapezoid(k * q1, k)
    m2_1 = np.trapezoid(k * q2, k)
    m1_2 = np.trapezoid(k**2 * q1, k)
    m2_2 = np.trapezoid(k**2 * q2, k)
    return float(np.abs(m1_1 - m2_1)), float(np.abs(m1_2 - m2_2))
