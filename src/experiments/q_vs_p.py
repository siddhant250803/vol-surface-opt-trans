"""Experiment B: Q vs P divergence, RV prediction.

Units convention:
- Returns: decimal (0.01 = 1%)
- RV_h: sum of r² over returns spanning h calendar days = horizon variance
- RV_ann: RV_h × (365 / span_days) = annualized variance
- atm_iv: SVI total variance w = IV² × tau (tau in years)
- IV²_ann: atm_iv / tau = annualized variance
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def run_q_vs_p_experiment(
    features: pl.DataFrame,
    distances: pl.DataFrame,
    options: pl.DataFrame,
    config: dict[str, Any],
) -> pl.DataFrame:
    """Add realized variance for RV−IV² analysis."""
    exp_cfg = config.get("experiments", {}).get("B_q_vs_p", {})
    if not exp_cfg.get("enabled", True):
        return features

    rv_horizons = exp_cfg.get("rv_horizon_days", [5, 7, 10, 21])

    if "spx_price" not in options.columns:
        return features

    prices = options.select(["date", "spx_price"]).unique().sort("date")
    prices = prices.with_columns(pl.col("spx_price").log().alias("log_p"))
    prices = prices.with_columns(pl.col("log_p").diff().alias("log_ret"))
    prices = prices.filter(pl.col("log_ret").is_not_null())

    dates = prices["date"].to_numpy()
    rets = prices["log_ret"].to_numpy()

    def _days_between(d0, d1) -> int:
        try:
            delta = d1 - d0
            if hasattr(delta, "days"):
                return int(delta.days)
            # numpy timedelta64
            return int(delta / np.timedelta64(1, "D"))
        except Exception:
            return 0

    def realized_var(start_idx: int, target_days: int) -> tuple[float, float]:
        """RV over returns spanning >= target_days. Returns (RV_horizon, span_days)."""
        if start_idx + 1 >= len(rets):
            return np.nan, np.nan
        d0 = dates[start_idx]
        rv = 0.0
        span = 0
        for i in range(start_idx, len(rets)):
            rv += rets[i] ** 2
            if i + 1 < len(dates):
                span = _days_between(d0, dates[i + 1])
            if span >= target_days:
                return float(rv), float(max(span, 1))
        return np.nan, np.nan

    date_to_idx = {str(d): i for i, d in enumerate(dates)}
    unique_dates = features["date"].unique().to_list()

    rv_dfs = []
    for h in rv_horizons:
        rv_vals = []
        span_vals = []
        for d in unique_dates:
            idx = date_to_idx.get(str(d), -1)
            if idx < 0:
                rv_vals.append(np.nan)
                span_vals.append(np.nan)
            else:
                rv_h, span = realized_var(idx, h)
                rv_vals.append(rv_h)
                span_vals.append(span)
        # RV_h = horizon variance. RV_ann = RV_h * (365/span) for same-scale as IV²_ann
        rv_ann = [
            rv * (365 / span) if not np.isnan(rv) and span and span > 0 else np.nan
            for rv, span in zip(rv_vals, span_vals)
        ]
        df = pl.DataFrame({
            "date": unique_dates,
            f"realized_var_{h}d": rv_vals,
            f"realized_var_{h}d_span_days": span_vals,
            f"realized_var_{h}d_ann": rv_ann,
        })
        rv_dfs.append(df.select(["date", f"realized_var_{h}d", f"realized_var_{h}d_ann"]))

    rv_merged = rv_dfs[0]
    for df in rv_dfs[1:]:
        rv_merged = rv_merged.join(df, on="date", how="left")
    # Prefer _ann for regression (same scale as IV²). Keep raw for diagnostics.
    features = features.join(rv_merged, on="date", how="left")

    return features
