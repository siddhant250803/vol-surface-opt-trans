"""Fitted call prices on k-grid from SVI."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm


def _svi_total_var(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    d = np.sqrt((k - m) ** 2 + sigma**2)
    return a + b * (rho * (k - m) + d)


def _bs_call_price(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Black-Scholes call price in log-moneyness space: C/F = N(d1) - exp(k)N(d2)."""
    sigma = np.sqrt(np.maximum(w, 1e-12))
    d1 = -k / sigma + 0.5 * sigma
    d2 = d1 - sigma
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)


def compute_call_prices(svi_results: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """Compute call prices on k-grid for each fitted slice."""
    k_cfg = config.get("density", {}).get("k_grid", {})
    n_points = k_cfg.get("n_points", 201)
    k_min = k_cfg.get("k_min", -0.5)
    k_max = k_cfg.get("k_max", 0.5)
    k_grid = np.linspace(k_min, k_max, n_points)

    rows = []
    for i in range(len(svi_results)):
        row = svi_results.row(i, named=True)
        a, b, rho, m, sigma = row["svi_a"], row["svi_b"], row["svi_rho"], row["svi_m"], row["svi_sigma"]
        if np.isnan(a) or row["fit_status"] == "fail":
            continue

        w = _svi_total_var(k_grid, a, b, rho, m, sigma)
        c = _bs_call_price(k_grid, w)

        rows.append({
            "date": row["date"],
            "exdate": row["exdate"],
            "tau_bucket": row["tau_bucket"],
            "target_tau_days": row["target_tau_days"],
            "forward": row["forward"],
            "tau": row["tau"],
            "spx_price": row["spx_price"],
            "k_grid": k_grid.tolist(),
            "call_prices": c.tolist(),
            "fit_status": row["fit_status"],
            "iv_rmse_vega_weighted": row["iv_rmse_vega_weighted"],
        })

    return pl.DataFrame(rows)
