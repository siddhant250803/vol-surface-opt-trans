"""SVI parameter-space distance: stable surface shift metric without PDF differentiation."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def svi_param_vector(row: dict[str, Any]) -> np.ndarray:
    """Extract θ = (a, b, ρ, m, σ) from SVI row."""
    return np.array([
        float(row.get("svi_a", np.nan)),
        float(row.get("svi_b", np.nan)),
        float(row.get("svi_rho", np.nan)),
        float(row.get("svi_m", np.nan)),
        float(row.get("svi_sigma", np.nan)),
    ])


def svi_param_distance(
    theta1: np.ndarray,
    theta2: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """L2 distance between SVI parameter vectors. Optional per-param weights."""
    if np.any(np.isnan(theta1)) or np.any(np.isnan(theta2)):
        return np.nan
    diff = theta1 - theta2
    if weights is not None:
        diff = diff * np.sqrt(weights)
    return float(np.linalg.norm(diff))


def compute_svi_param_distances(
    svi_results: pl.DataFrame,
    config: dict[str, Any],
) -> pl.DataFrame:
    """Compute svi_param_dist_prev = distance to previous day's SVI params per (date, tau_bucket)."""
    dist_cfg = config.get("distances", {}).get("svi_params", {})
    weights = dist_cfg.get("param_weights")
    if weights is not None:
        weights = np.array(weights)
    elif dist_cfg.get("normalize", True):
        weights = np.array([10.0, 5.0, 2.0, 5.0, 5.0])
    else:
        weights = np.ones(5)

    svi_sorted = svi_results.filter(
        pl.col("fit_status") == "ok"
    ).sort(["tau_bucket", "date"])

    if svi_sorted.is_empty():
        return pl.DataFrame()

    rows = []
    for tau_bucket in svi_sorted["tau_bucket"].unique().to_list():
        sub = svi_sorted.filter(pl.col("tau_bucket") == tau_bucket)
        dates = sub["date"].to_list()
        for i in range(len(sub)):
            row = sub.row(i, named=True)
            theta = svi_param_vector(row)
            dist_prev = np.nan
            if i > 0:
                prev_row = sub.row(i - 1, named=True)
                theta_prev = svi_param_vector(prev_row)
                dist_prev = svi_param_distance(theta, theta_prev, weights)
            rows.append({
                "date": row["date"],
                "tau_bucket": tau_bucket,
                "svi_param_dist_prev": dist_prev,
                "svi_a": row.get("svi_a"),
                "svi_b": row.get("svi_b"),
                "svi_rho": row.get("svi_rho"),
                "svi_m": row.get("svi_m"),
                "svi_sigma": row.get("svi_sigma"),
            })

    return pl.DataFrame(rows)
