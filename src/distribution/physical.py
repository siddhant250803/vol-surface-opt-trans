"""Physical distribution P: bootstrap from historical returns."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _get_log_returns(options: pl.DataFrame) -> pl.DataFrame:
    """Extract daily log returns from SPX prices."""
    if "spx_price" not in options.columns:
        return pl.DataFrame()

    prices = options.select(["date", "spx_price"]).unique().sort("date")
    prices = prices.with_columns(pl.col("spx_price").log().alias("log_price"))
    prices = prices.with_columns(pl.col("log_price").diff().alias("log_return"))
    return prices.filter(pl.col("log_return").is_not_null())


def estimate_physical_density(
    options: pl.DataFrame,
    rates: pl.DataFrame,
    config: dict[str, Any],
    target_dates: list | None = None,
) -> pl.DataFrame:
    """Estimate P for each (date, tau) via bootstrap of historical returns."""
    phys_cfg = config.get("physical", {})
    window_days = phys_cfg.get("window_days", 252)
    bootstrap_n = phys_cfg.get("bootstrap_n", 1000)

    returns_df = _get_log_returns(options)
    if returns_df.is_empty():
        return pl.DataFrame()

    returns = returns_df["log_return"].to_numpy().astype(float)
    dates = returns_df["date"].to_numpy()

    k_cfg = config.get("density", {}).get("k_grid", {})
    n_points = k_cfg.get("n_points", 201)
    k_min = k_cfg.get("k_min", -0.5)
    k_max = k_cfg.get("k_max", 0.5)
    k_grid = np.linspace(k_min, k_max, n_points)

    unique_dates = target_dates if target_dates is not None else options.select("date").unique().sort("date")["date"].to_list()
    buckets = config.get("tau_buckets", [])
    if not buckets:
        return pl.DataFrame()
    tau_buckets = [{"tau_bucket": b.get("label", str(b["target_days"])), "target_tau_days": b["target_days"]} for b in buckets]

    rows = []
    for date_val in unique_dates:
        idx = np.where(dates <= date_val)[0]
        if len(idx) < window_days:
            continue
        window_returns = returns[idx[-window_days]:]

        for tb_row in tau_buckets:
            tau_bucket = tb_row.get("tau_bucket", str(tb_row["target_tau_days"]))
            target_tau = tb_row["target_tau_days"]
            tau_years = target_tau / 365.0
            n_steps = max(1, int(target_tau))

            boot_x = []
            for _ in range(bootstrap_n):
                idx_boot = np.random.choice(len(window_returns), size=n_steps, replace=True)
                cum_ret = np.sum(window_returns[idx_boot])
                boot_x.append(cum_ret)
            boot_x = np.array(boot_x)

            hist, bin_edges = np.histogram(boot_x, bins=n_points, range=(k_min, k_max), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            p_density = hist.astype(float)
            integral = np.trapezoid(p_density, bin_centers)
            if integral > 1e-10:
                p_density = p_density / integral

            # Use bin_centers (where density is defined), not k_grid
            rows.append({
                "date": date_val,
                "tau_bucket": tau_bucket,
                "target_tau_days": target_tau,
                "k_grid": bin_centers.tolist(),
                "p_density": p_density.tolist(),
                "p_integral": float(np.trapezoid(p_density, bin_centers)),
            })

    return pl.DataFrame(rows)
