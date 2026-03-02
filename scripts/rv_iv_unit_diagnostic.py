#!/usr/bin/env python3
"""RV–IV² unit diagnostic: verify returns, RV, IV on same scale. Run after pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from pipeline.config import load_config, compute_config_hash


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "configs" / "base.yaml")
    config_hash = compute_config_hash(config)

    opts = pl.read_parquet(project_root / "data" / "processed" / config_hash / "options_clean.parquet")
    feat = pl.read_parquet(project_root / "outputs" / "features" / f"features_{config_hash}.parquet")

    # --- 1. Returns: decimal or percent? ---
    prices = opts.select(["date", "spx_price"]).unique().sort("date")
    prices = prices.with_columns(pl.col("spx_price").log().alias("log_p"))
    prices = prices.with_columns(pl.col("log_p").diff().alias("log_ret"))
    prices = prices.filter(pl.col("log_ret").is_not_null())

    rets = prices["log_ret"].to_numpy()
    dates = prices["date"].to_numpy()
    date_to_idx = {str(d): i for i, d in enumerate(dates)}

    print("=" * 70)
    print("RV–IV² UNIT DIAGNOSTIC")
    print("=" * 70)

    print("\n1. RETURNS (log_ret = diff(log(price)))")
    print("   Expected: decimal (e.g. 0.01 = 1%). If percent, values ~1.0.")
    sample_rets = rets[:20]
    print(f"   First 20 returns: min={sample_rets.min():.6f}, max={sample_rets.max():.6f}, mean_abs={np.abs(sample_rets).mean():.6f}")
    if np.abs(sample_rets).mean() > 0.1:
        print("   *** RED FLAG: Returns look like percent (mean_abs > 0.1). Should be ~0.01. ***")
    else:
        print("   OK: Returns appear to be in decimal.")

    # --- 2. RV for one date (span-aware) ---
    rv_col = "realized_var_7d_ann" if "realized_var_7d_ann" in feat.columns else "realized_var_7d"
    sample_row = feat.filter(pl.col(rv_col).is_not_null()).row(0, named=True)
    sample_date = sample_row["date"]
    idx = date_to_idx.get(str(sample_date))
    if idx is not None and idx + 1 < len(rets):
        d0 = dates[idx]
        rv_h, span, n_ret = 0.0, 0, 0
        for i in range(idx, len(rets)):
            rv_h += rets[i] ** 2
            n_ret += 1
            if i + 1 < len(dates):
                delta = dates[i + 1] - d0
                try:
                    span = int(delta.days) if hasattr(delta, "days") else int(delta / np.timedelta64(1, "D"))
                except Exception:
                    span = 0
            if span >= 7:
                break
        rv_ann = rv_h * (365 / max(span, 1)) if span >= 7 else np.nan
        print(f"\n2. RV_7d for date {sample_date} (span-aware)")
        print(f"   N returns used: {n_ret}, span: {span} calendar days")
        print(f"   RV_h = sum(r²) = {rv_h:.8f}  [horizon variance]")
        if span >= 7:
            print(f"   RV_ann = RV_h × (365/{span}) = {rv_ann:.6f}  [annualized variance]")
            print(f"   sqrt(RV_ann) = {np.sqrt(rv_ann):.4f}  [annualized vol, expect 0.15–0.35]")
            if np.sqrt(rv_ann) > 1.0:
                print("   *** RED FLAG: Annualized vol > 100%. Returns may be in percent. ***")

    # --- 3. IV from features ---
    row = feat.filter(pl.col("date") == sample_date).row(0, named=True)
    atm_iv = row.get("atm_iv")
    if atm_iv is not None and not np.isnan(atm_iv):
        tau_7d = 7 / 365
        # atm_iv in features = SVI total variance w at ATM (w = iv² * tau)
        # So iv_ann² = w / tau = atm_iv / tau_7d
        iv_sq_ann = atm_iv / tau_7d
        iv_ann = np.sqrt(iv_sq_ann)
        print(f"\n3. IV for date {sample_date}")
        print(f"   atm_iv (SVI w at ATM) = {atm_iv:.6f}  [total variance over 7D]")
        print(f"   IV²_ann = atm_iv / tau = atm_iv / (7/365) = {iv_sq_ann:.6f}")
        print(f"   IV_ann = sqrt(IV²) = {iv_ann:.4f}  [expect 0.15–0.35 decimal, or 15–35%]")
        if iv_ann > 1.0:
            print("   *** RED FLAG: IV > 1 (100%). atm_iv may be in wrong units. ***")

    # --- 4. RV − IV² comparison ---
    if "realized_var_7d" in feat.columns or "realized_var_7d_ann" in feat.columns:
        rv_c = "realized_var_7d_ann" if "realized_var_7d_ann" in feat.columns else "realized_var_7d"
        sub = feat.filter(
            pl.col(rv_c).is_not_null()
            & pl.col("atm_iv").is_not_null()
            & pl.col("atm_iv").gt(0)
        )
        if len(sub) > 0:
            tau_7d = 7 / 365
            rv_ann = sub[rv_c] if rv_c == "realized_var_7d_ann" else sub[rv_c] * (365 / 7)
            iv_sq = sub["atm_iv"] / tau_7d
            rv_iv = rv_ann - iv_sq
            rv_finite = rv_ann.to_numpy()
            rv_finite = rv_finite[np.isfinite(rv_finite)]
            rv_iv_finite = rv_iv.to_numpy()
            rv_iv_finite = rv_iv_finite[np.isfinite(rv_iv_finite)]
            print(f"\n4. RV − IV² (annualized, 7D horizon)")
            print(f"   Mean RV_ann: {float(np.mean(rv_finite)):.6f}" if len(rv_finite) > 0 else "   Mean RV_ann: N/A")
            print(f"   Mean IV²_ann: {float(iv_sq.mean()):.6f}")
            print(f"   Mean(RV − IV²): {float(np.mean(rv_iv_finite)):.6f}" if len(rv_iv_finite) > 0 else "   Mean(RV − IV²): N/A")
            print(f"   Std(RV − IV²): {float(np.std(rv_iv_finite)):.6f}" if len(rv_iv_finite) > 1 else "   Std(RV − IV²): N/A")
            if rv_iv.mean() > 0.5:
                print("   *** RED FLAG: Mean(RV−IV²) > 0.5 suggests unit mismatch. Typical: -0.02 to 0.05 ***")

    print("\n5. CONVENTION (documented)")
    print("   - Returns: decimal (0.01 = 1%)")
    print("   - RV_h = sum(r²) over h days = horizon variance")
    print("   - RV_ann = RV_h × (365/h) = annualized variance")
    print("   - atm_iv = SVI total variance w = IV² × tau (tau in years)")
    print("   - IV²_ann = atm_iv / tau = annualized variance")
    print("   - Both RV_ann and IV²_ann in same units → RV − IV² comparable")


if __name__ == "__main__":
    main()
