"""Experiment A: Q_t vs Q_{t-1}, ATM IV, skew, SVI param distance."""

from __future__ import annotations

from typing import Any

import polars as pl

from src.distances.svi_params import compute_svi_param_distances


def run_q_vs_q_experiment(
    distances: pl.DataFrame,
    svi_results: pl.DataFrame,
    config: dict[str, Any],
) -> pl.DataFrame:
    """Add ATM IV, skew, SVI param distance, and their 1d changes to features. Mark valid_row."""
    exp_cfg = config.get("experiments", {}).get("A_q_vs_q", {})
    if not exp_cfg.get("enabled", True):
        return distances

    fit_cfg = config.get("fit", {})
    iv_rmse_max = fit_cfg.get("iv_rmse_max", 0.02)
    mass_err_max = fit_cfg.get("density_mass_error_max", 0.01)

    svi_param_dist = compute_svi_param_distances(svi_results, config)
    svi_by = svi_results.group_by(["date", "tau_bucket"]).first()
    svi_cols = ["date", "tau_bucket", "svi_a", "svi_b", "svi_rho", "svi_m", "svi_sigma", "iv_rmse_vega_weighted"]
    svi_flat = svi_by.select([c for c in svi_cols if c in svi_by.columns])

    atm_iv = svi_flat.with_columns(
        (pl.col("svi_a") + pl.col("svi_b") * pl.col("svi_sigma") * (pl.lit(1) - pl.col("svi_rho")**2).sqrt()).alias("atm_iv")
    )
    skew = svi_flat.with_columns(
        (pl.col("svi_b") * pl.col("svi_rho") / (pl.col("svi_a") + pl.col("svi_b") * pl.col("svi_sigma") * (pl.lit(1) - pl.col("svi_rho")**2).sqrt())).alias("skew")
    )

    features = distances.join(
        svi_flat.select(["date", "tau_bucket", "iv_rmse_vega_weighted"]),
        on=["date", "tau_bucket"],
        how="left",
    ).join(
        atm_iv.select(["date", "tau_bucket", "atm_iv"]),
        on=["date", "tau_bucket"],
        how="left",
    ).join(
        skew.select(["date", "tau_bucket", "skew"]),
        on=["date", "tau_bucket"],
        how="left",
    )
    if not svi_param_dist.is_empty():
        features = features.join(
            svi_param_dist.select(["date", "tau_bucket", "svi_param_dist_prev"]),
            on=["date", "tau_bucket"],
            how="left",
        )

    features = features.sort(["date", "tau_bucket"])
    features = features.with_columns(
        pl.col("atm_iv").shift(1).over("tau_bucket").alias("atm_iv_prev"),
        pl.col("skew").shift(1).over("tau_bucket").alias("skew_prev"),
    )
    features = features.with_columns(
        (pl.col("atm_iv") - pl.col("atm_iv_prev")).alias("atm_iv_change_1d"),
        (pl.col("skew") - pl.col("skew_prev")).alias("skew_change_1d"),
    )

    features = features.with_columns(
        (pl.col("iv_rmse_vega_weighted").fill_null(999) <= iv_rmse_max).alias("fit_ok")
    )
    features = features.with_columns(
        (pl.col("q_integral").is_between(1 - mass_err_max, 1 + mass_err_max)).fill_null(False).alias("density_ok")
    )
    not_invalid = (pl.col("invalid_density") == 0) | pl.col("invalid_density").is_null()
    features = features.with_columns(
        (pl.col("fit_ok") & pl.col("density_ok") & not_invalid).alias("valid_row")
    )

    return features
