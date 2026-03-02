"""Constant-maturity Q: interpolate across expiries to exactly target_tau_days."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl

from src.distribution.risk_neutral import recover_risk_neutral_density


def _interpolate_svi_params(
    tau_vals: np.ndarray,
    params: list[dict[str, float]],
    target_tau: float,
) -> tuple[float, float, float, float, float] | None:
    """Linear interpolation of SVI params in tau. Returns (a,b,rho,m,sigma) or None."""
    if len(tau_vals) == 0:
        return None
    order = np.argsort(tau_vals)
    tau_s = tau_vals[order]
    if target_tau < tau_s[0] or target_tau > tau_s[-1]:
        return None
    p_ordered = [params[i] for i in order]
    a_vals = np.array([p["a"] for p in p_ordered])
    b_vals = np.array([p["b"] for p in p_ordered])
    rho_vals = np.array([p["rho"] for p in p_ordered])
    m_vals = np.array([p["m"] for p in p_ordered])
    sigma_vals = np.array([p["sigma"] for p in p_ordered])
    return (
        float(np.interp(target_tau, tau_s, a_vals)),
        float(np.interp(target_tau, tau_s, b_vals)),
        float(np.interp(target_tau, tau_s, rho_vals)),
        float(np.interp(target_tau, tau_s, m_vals)),
        float(np.interp(target_tau, tau_s, sigma_vals)),
    )


def build_constant_maturity_q(
    svi_results: pl.DataFrame,
    config: dict[str, Any],
    tau_buckets: list[str] | None = None,
) -> pl.DataFrame:
    """Build Q at constant maturity by interpolating SVI across expiries.

    For each (date, tau_bucket), if multiple expiries exist, interpolate SVI params
    to target_tau_days, then compute C and Q. Q_prev = Q_7D(t-1) uses this.
    """
    buckets = config.get("tau_buckets", [])
    if tau_buckets:
        buckets = [b for b in buckets if b.get("label") in tau_buckets]
    if not buckets:
        return pl.DataFrame()

    k_cfg = config.get("density", {}).get("k_grid", {})
    n_points = k_cfg.get("n_points", 201)
    k_min = k_cfg.get("k_min", -0.5)
    k_max = k_cfg.get("k_max", 0.5)
    k_grid = np.linspace(k_min, k_max, n_points)

    from src.surface.call_prices import _svi_total_var
    from src.surface.call_prices import _bs_call_price

    rows = []
    for bucket in buckets:
        target_days = bucket["target_days"]
        label = bucket.get("label", str(target_days))
        target_tau = target_days / 365.0

        sub = svi_results.filter(
            (pl.col("tau_bucket") == label) & (pl.col("fit_status") == "ok")
        )
        if sub.is_empty():
            continue

        for date_val in sub["date"].unique().to_list():
            day_sub = sub.filter(pl.col("date") == date_val)
            if len(day_sub) == 1:
                row = day_sub.row(0, named=True)
                tau_actual = float(row["tau"])
                if abs(tau_actual - target_tau) > 2 / 365.0:
                    continue
                a, b, rho, m, sigma = row["svi_a"], row["svi_b"], row["svi_rho"], row["svi_m"], row["svi_sigma"]
            else:
                tau_vals = day_sub["tau"].to_numpy().astype(float)
                params = [
                    {"a": r["svi_a"], "b": r["svi_b"], "rho": r["svi_rho"], "m": r["svi_m"], "sigma": r["svi_sigma"]}
                    for r in day_sub.iter_rows(named=True)
                ]
                interp = _interpolate_svi_params(tau_vals, params, target_tau)
                if interp is None:
                    continue
                a, b, rho, m, sigma = interp

            forward = float(day_sub["forward"].median())
            w = _svi_total_var(k_grid, a, b, rho, m, sigma)
            c = _bs_call_price(k_grid, w)

            exdate_val = date_val + timedelta(days=target_days) if hasattr(date_val, "__add__") else date_val
            rows.append({
                "date": date_val,
                "exdate": exdate_val,
                "tau_bucket": label,
                "target_tau_days": target_days,
                "forward": forward,
                "tau": target_tau,
                "spx_price": forward,
                "k_grid": k_grid.tolist(),
                "call_prices": c.tolist(),
                "fit_status": "ok",
                "constant_maturity": True,
            })

    if not rows:
        return pl.DataFrame()

    call_df = pl.DataFrame(rows)
    q_df = recover_risk_neutral_density(call_df, config)
    return q_df.with_columns(pl.lit(True).alias("constant_maturity"))
