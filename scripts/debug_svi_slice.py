#!/usr/bin/env python3
"""Debug SVI fit for one slice: print params, w_mkt vs w_fit, IV overlay."""

import sys
from pathlib import Path

import numpy as np
import polars as pl

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from pipeline.config import load_config
from src.data.load_options import load_options
from src.data.load_rates import load_rates
from src.data.clean import clean_options
from src.surface.tau_mapping import map_tau_buckets
from src.surface.svi_fit import fit_svi_slice, _svi_total_var, _iv_from_total_var


def main():
    config = load_config(project_root / "configs/tests/golden.yaml")
    opts_path = project_root / config["data"]["options_path"]
    rates_path = project_root / config["data"]["rates_path"]

    options = load_options(str(opts_path), config)
    rates = load_rates(str(rates_path), config)
    options_clean = clean_options(options, config)
    tau_mapped = map_tau_buckets(options_clean, config)

    df = tau_mapped.filter(
        (pl.col("date").cast(str) == "2017-05-26") & (pl.col("tau_bucket") == "7D")
    )
    if df.is_empty():
        df = tau_mapped.filter(pl.col("tau_bucket") == "7D").head(1)
        keys = tau_mapped.filter(pl.col("tau_bucket") == "7D").select(["date", "exdate", "tau_bucket", "target_tau_days"]).unique().head(1)
    else:
        keys = df.select(["date", "exdate", "tau_bucket", "target_tau_days"]).unique()

    row = keys.row(0, named=True)
    date_val, exdate_val, tau_bucket, target_tau = row["date"], row["exdate"], row["tau_bucket"], row["target_tau_days"]
    sub = tau_mapped.filter(
        (pl.col("date") == date_val) & (pl.col("exdate") == exdate_val) &
        (pl.col("tau_bucket") == tau_bucket) & (pl.col("target_tau_days") == target_tau)
    )

    forward = sub["spx_price"].median()
    strike = sub["strike_price"].to_numpy() / 1000.0
    k = np.log(strike / forward)
    iv = sub["impl_volatility"].to_numpy().astype(float)
    vega = sub["vega"].to_numpy() if "vega" in sub.columns else None
    if vega is not None:
        vega = np.maximum(vega.astype(float), 1e-6)

    tau_days = (exdate_val - date_val).days if hasattr(exdate_val, "days") else target_tau
    tau = tau_days / 365.0

    valid = np.isfinite(iv) & (iv > 0) & np.isfinite(k)
    k, iv = k[valid], iv[valid]
    vega = vega[valid] if vega is not None else None

    fit = fit_svi_slice(k, iv, vega, tau, config)

    w_mkt = (iv ** 2) * tau
    w_fit = _svi_total_var(k, fit["svi_a"], fit["svi_b"], fit["svi_rho"], fit["svi_m"], fit["svi_sigma"])
    iv_fit = _iv_from_total_var(w_fit, tau)

    print("=" * 60)
    print(f"SVI Debug: {date_val} {tau_bucket}")
    print("=" * 60)
    print("Calibrated params: a={:.6f} b={:.6f} rho={:.6f} m={:.6f} sigma={:.6f}".format(
        fit["svi_a"], fit["svi_b"], fit["svi_rho"], fit["svi_m"], fit["svi_sigma"]))
    print("Mean market total variance w_mkt:  {:.6e}".format(np.mean(w_mkt)))
    print("Mean fitted total variance w_fit:  {:.6e}".format(np.mean(w_fit)))
    print("Min/max w_mkt:  {:.6e} / {:.6e}".format(np.min(w_mkt), np.max(w_mkt)))
    print("Min/max w_fit:  {:.6e} / {:.6e}".format(np.min(w_fit), np.max(w_fit)))
    print("Mean market IV:  {:.4f}".format(np.mean(iv)))
    print("Mean fitted IV:  {:.4f}".format(np.mean(iv_fit)))
    print("fit_status:", fit["fit_status"])
    print("iv_rmse_vega_weighted:", fit["iv_rmse_vega_weighted"])


if __name__ == "__main__":
    main()
