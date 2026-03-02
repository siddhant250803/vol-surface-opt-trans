"""Units and invariants: IV decimal, w = iv²*T (years)."""

from __future__ import annotations

import numpy as np


def assert_iv_bounds(iv: np.ndarray, label: str = "IV") -> None:
    """IV must be decimal (0.01 = 1%), in (0, 3)."""
    if np.any(iv <= 0) or np.any(iv >= 3):
        bad = np.where((iv <= 0) | (iv >= 3))[0]
        raise ValueError(f"{label} out of (0, 3): min={iv.min():.4f} max={iv.max():.4f} at {len(bad)} pts")


def assert_total_var_bounds(w: np.ndarray, label: str = "w") -> None:
    """Total variance w must be in (0, 1)."""
    if np.any(w <= 0) or np.any(w >= 1):
        raise ValueError(f"{label} out of (0, 1): min={w.min():.6e} max={w.max():.6e}")


def assert_q_integral(integral: float, tol: float = 1e-3, label: str = "Q") -> None:
    """∫q(x)dx must be within tol of 1."""
    if abs(integral - 1.0) >= tol:
        raise ValueError(f"{label} integral={integral:.6f} not in [1±{tol}]")


def w_from_iv(iv: np.ndarray, tau_years: float) -> np.ndarray:
    """Total variance w = iv² * tau. IV decimal, tau in years."""
    return (iv ** 2) * tau_years


def iv_from_w(w: np.ndarray, tau_years: float) -> np.ndarray:
    """IV = sqrt(w / tau). Returns decimal."""
    with np.errstate(divide="ignore", invalid="ignore"):
        iv = np.sqrt(np.maximum(w / tau_years, 1e-12))
    return np.where(np.isfinite(iv), iv, np.nan)
