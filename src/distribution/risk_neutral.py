"""Recover risk-neutral Q from call prices via Breeden-Litzenberger: q(K) = e^{rT} ∂²C/∂K²."""

from __future__ import annotations

from typing import Any

import numpy as np

import polars as pl

from src.utils.units import assert_q_integral


def _first_derivative_central(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Central difference for dy/dx."""
    d = np.zeros_like(y, dtype=float)
    d[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    d[0] = (y[1] - y[0]) / (x[1] - x[0]) if len(x) > 1 else 0.0
    d[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2]) if len(x) > 1 else 0.0
    return d


def _second_derivative_central(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Central difference for d²y/dx²."""
    d2 = np.zeros_like(y, dtype=float)
    dx = np.diff(x)
    if len(dx) < 2:
        return d2
    dx_lo, dx_hi = dx[:-1], dx[1:]
    d2[1:-1] = 2 * (y[2:] - 2 * y[1:-1] + y[:-2]) / (dx_lo * (dx_lo + dx_hi))
    d2[0], d2[-1] = d2[1], d2[-2]
    return d2


def recover_risk_neutral_density(call_prices_df: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """Recover Q via Breeden-Litzenberger: q(K) = e^{rT} ∂²C/∂K².

    Uses direct second derivative (no CDF clipping). In log-moneyness: q_k(k) = e^{rT} [∂²c/∂k² - ∂c/∂k] / exp(k).
    """
    density_cfg = config.get("density", {})
    clip_neg = density_cfg.get("clip_negative", False)  # diagnostic only, not main step
    renormalize = density_cfg.get("renormalize", True)
    assert_units = density_cfg.get("assert_units", config.get("fit", {}).get("assert_units", False))
    fit_cfg = config.get("fit", {})
    mass_err_max = fit_cfg.get("density_mass_error_max", 0.01)
    r_default = config.get("density", {}).get("r_default", 0.0)

    rows = []
    for i in range(len(call_prices_df)):
        row = call_prices_df.row(i, named=True)
        k = np.array(row["k_grid"], dtype=float)
        c = np.array(row["call_prices"], dtype=float)
        F = float(row["forward"])
        tau = float(row["tau"])
        r = float(row.get("r", r_default))

        if len(k) < 5:
            continue

        # q_k(k) = e^{rT} [∂²c/∂k² - ∂c/∂k] / exp(k)  (Breeden-Litzenberger in log-moneyness)
        dc_dk = _first_derivative_central(c, k)
        d2c_dk2 = _second_derivative_central(c, k)
        q_raw = np.exp(r * tau) * (d2c_dk2 - dc_dk) / np.exp(k)
        q_raw = np.nan_to_num(q_raw, nan=0.0, posinf=0.0, neginf=0.0)

        clip_frac = np.mean(q_raw < 0)
        q = np.maximum(q_raw, 0.0)  # PDF must be non-negative

        if renormalize:
            trapz = np.trapezoid(q, k)
            if trapz > 1e-10:
                q = q / trapz

        # CDF from integrated PDF (for compatibility)
        cdf = np.concatenate([[0], np.cumsum(q[:-1] * np.diff(k))])
        cdf = np.minimum(np.maximum(cdf, 0.0), 1.0)

        integral = float(np.trapezoid(q, k))
        mass_outside = 1.0 - integral if integral < 1 else 0.0
        invalid_density = abs(integral - 1.0) > mass_err_max or np.any(np.isnan(q))
        if assert_units:
            assert_q_integral(integral, tol=1e-3, label="Q")

        rows.append({
            "date": row["date"],
            "exdate": row["exdate"],
            "tau_bucket": row["tau_bucket"],
            "target_tau_days": row["target_tau_days"],
            "forward": row["forward"],
            "tau": row["tau"],
            "spx_price": row["spx_price"],
            "k_grid": k.tolist(),
            "q_density": q.tolist(),
            "q_cdf": cdf.tolist(),
            "q_integral": integral,
            "q_neg_clip_frac": clip_frac,
            "q_mass_outside_grid": mass_outside,
            "invalid_density": invalid_density,
            "fit_status": row["fit_status"],
        })

    return pl.DataFrame(rows)
