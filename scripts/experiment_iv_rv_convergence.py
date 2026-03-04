#!/usr/bin/env python3
"""IV surface dynamics and RV-IV convergence as options approach expiry.

Experiments:
1. IV surface over time: ATM IV, skew, SVI param distance by tau bucket
2. RV/IV² realization ratio by horizon (5d, 7d, 10d, 21d)
3. Convergence: RV vs IV² by days-to-expiry (how RV and IV converge as options approach expiry)

Units: atm_iv = total variance w = IV² × tau; IV²_ann = atm_iv / tau.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from pipeline.config import load_config, compute_config_hash


def _days_between(d0, d1) -> int:
    try:
        delta = d1 - d0
        if hasattr(delta, "days"):
            return int(delta.days)
        return int(delta / np.timedelta64(1, "D"))
    except Exception:
        return 0


def _realized_var(prices: pl.DataFrame, start_date, end_date) -> tuple[float, float]:
    """RV over returns from start_date to end_date. Returns (RV_horizon, span_days)."""
    dates = prices["date"].to_list()
    rets = prices["log_ret"].to_numpy()
    date_to_idx = {str(d): i for i, d in enumerate(dates)}

    start_idx = date_to_idx.get(str(start_date), -1)
    end_idx = date_to_idx.get(str(end_date), -1)
    if start_idx < 0 or end_idx < 0 or start_idx >= end_idx:
        return np.nan, np.nan

    rv = 0.0
    for i in range(start_idx, end_idx):
        rv += rets[i] ** 2
    span = _days_between(dates[start_idx], dates[end_idx])
    return float(rv), float(max(span, 1))


def run_iv_surface_dynamics(svi: pl.DataFrame, out_dir: Path, dpi: int = 150) -> None:
    """Track IV surface metrics over time: ATM IV, skew, term structure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    svi_by = svi.group_by(["date", "tau_bucket"]).first()
    svi_cols = ["date", "tau_bucket", "svi_a", "svi_b", "svi_rho", "svi_m", "svi_sigma"]
    svi_flat = svi_by.select([c for c in svi_cols if c in svi_by.columns])
    if svi_flat.is_empty():
        return

    atm_iv = svi_flat.with_columns(
        (pl.col("svi_a") + pl.col("svi_b") * pl.col("svi_sigma")
         * (pl.lit(1) - pl.col("svi_rho") ** 2).sqrt()).alias("atm_iv")
    )
    skew = svi_flat.with_columns(
        (pl.col("svi_b") * pl.col("svi_rho")
         / (pl.col("svi_a") + pl.col("svi_b") * pl.col("svi_sigma")
            * (pl.lit(1) - pl.col("svi_rho") ** 2).sqrt())).alias("skew")
    )

    buckets = sorted(atm_iv["tau_bucket"].unique().to_list())
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for b in buckets:
        sub = atm_iv.filter(pl.col("tau_bucket") == b).sort("date")
        if sub.is_empty():
            continue
        axes[0].plot(sub["date"].to_list(), sub["atm_iv"].to_list(), label=b, alpha=0.8)
    axes[0].set_ylabel("ATM total variance w")
    axes[0].set_title("IV surface over time: ATM total variance by tau bucket")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for b in buckets:
        sub = skew.filter(pl.col("tau_bucket") == b).sort("date")
        if sub.is_empty():
            continue
        axes[1].plot(sub["date"].to_list(), sub["skew"].to_list(), label=b, alpha=0.8)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Skew")
    axes[1].set_title("IV surface over time: Skew by tau bucket")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "iv_surface_over_time.png", dpi=dpi)
    plt.close(fig)


def run_rv_iv_by_horizon(features: pl.DataFrame, out_dir: Path, dpi: int = 150) -> pl.DataFrame:
    """RV/IV² realization ratio by horizon. Uses 7D bucket IV for 7d RV, etc."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = [5, 7, 10, 21]
    tau_map = {7: "7D", 14: "14D", 30: "30D", 60: "60D"}
    # Map horizon to nearest tau bucket
    h_to_bucket = {5: "7D", 7: "7D", 10: "14D", 21: "30D"}

    rows = []
    for h in horizons:
        rv_col = f"realized_var_{h}d_ann"
        if rv_col not in features.columns:
            continue
        bucket = h_to_bucket.get(h, "7D")
        tau_years = h / 365.0
        sub = features.filter(
            (pl.col("tau_bucket") == pl.lit(bucket))
            & pl.col("atm_iv").is_not_null()
            & pl.col("atm_iv").gt(0)
            & pl.col(rv_col).is_not_null()
            & pl.col(rv_col).is_finite()
        )
        if sub.is_empty():
            continue
        iv_sq = sub["atm_iv"] / tau_years
        rv = sub[rv_col]
        ratio = rv / iv_sq
        ratio_clean = ratio.filter(ratio.is_between(0.1, 3.0))
        if ratio_clean.len() < 5:
            continue
        rows.append({
            "horizon_days": h,
            "tau_bucket": bucket,
            "mean_rv_iv_ratio": float(ratio_clean.mean()),
            "std_rv_iv_ratio": float(ratio_clean.std()),
            "median_rv_iv_ratio": float(ratio_clean.median()),
            "n": ratio_clean.len(),
        })

    if not rows:
        return pl.DataFrame()

    tbl = pl.DataFrame(rows)
    print("\n" + "=" * 60)
    print("RV/IV² realization ratio by horizon")
    print("=" * 60)
    print(tbl)
    print("(Ratio < 1: IV rich vs RV; Ratio > 1: IV cheap vs RV)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(r["horizon_days"]) + "d" for r in rows], [r["mean_rv_iv_ratio"] for r in rows],
           yerr=[r["std_rv_iv_ratio"] for r in rows], capsize=4, alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Mean RV/IV²")
    ax.set_title("RV/IV² realization ratio by horizon (convergence proxy)")
    fig.tight_layout()
    fig.savefig(out_dir / "rv_iv_ratio_by_horizon.png", dpi=dpi)
    plt.close(fig)
    return tbl


def run_convergence_by_dte(
    svi: pl.DataFrame,
    options: pl.DataFrame,
    out_dir: Path,
    dpi: int = 150,
) -> pl.DataFrame:
    """RV vs IV² by days-to-expiry. As options approach expiry, how does RV converge to IV?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "spx_price" not in options.columns:
        print("No spx_price in options; skipping convergence-by-DTE.")
        return pl.DataFrame()

    prices = options.select(["date", "spx_price"]).unique().sort("date")
    prices = prices.with_columns(pl.col("spx_price").log().alias("log_p"))
    prices = prices.with_columns(pl.col("log_p").diff().alias("log_ret"))
    prices = prices.filter(pl.col("log_ret").is_not_null())

    svi_atm = svi.with_columns(
        (pl.col("svi_a") + pl.col("svi_b") * pl.col("svi_sigma")
         * (pl.lit(1) - pl.col("svi_rho") ** 2).sqrt()).alias("atm_iv")
    )
    svi_atm = svi_atm.with_columns(
        (pl.col("exdate") - pl.col("date")).dt.total_days().alias("tau_days")
    )

    rows = []
    for row in svi_atm.iter_rows(named=True):
        date_val = row["date"]
        exdate_val = row["exdate"]
        tau_days = row["tau_days"]
        atm_iv = row["atm_iv"]
        tau = row["tau"]
        if tau_days is None or tau_days < 1 or tau_days > 90 or atm_iv is None or atm_iv <= 0:
            continue
        iv_sq_ann = atm_iv / tau
        rv_h, span = _realized_var(prices, date_val, exdate_val)
        if np.isnan(rv_h) or span <= 0:
            continue
        rv_ann = rv_h * (365 / span)
        ratio = rv_ann / iv_sq_ann
        if not (0.1 < ratio < 3.0):
            continue
        rows.append({
            "date": date_val,
            "exdate": exdate_val,
            "tau_days": tau_days,
            "iv_sq_ann": iv_sq_ann,
            "rv_ann": rv_ann,
            "rv_iv_ratio": ratio,
        })

    if not rows:
        print("No valid (date, exdate) pairs for convergence.")
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    by_dte = df.group_by("tau_days").agg([
        pl.col("rv_iv_ratio").mean().alias("mean_rv_iv_ratio"),
        pl.col("rv_iv_ratio").std().alias("std_rv_iv_ratio"),
        pl.col("rv_iv_ratio").median().alias("median_rv_iv_ratio"),
        pl.len().alias("n"),
    ]).sort("tau_days")

    print("\n" + "=" * 60)
    print("RV/IV² by days-to-expiry (convergence as options approach expiry)")
    print("=" * 60)
    print(by_dte)
    print("(Lower tau_days = closer to expiry; ratio → 1 suggests convergence)")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    dtes = by_dte["tau_days"].to_list()
    means = by_dte["mean_rv_iv_ratio"].to_list()
    stds = by_dte["std_rv_iv_ratio"].to_list()
    axes[0].bar([str(d) for d in dtes], means, yerr=stds, capsize=4, alpha=0.8)
    axes[0].axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Mean RV/IV²")
    axes[0].set_title("RV/IV² by days-to-expiry")
    axes[0].grid(True, alpha=0.3)

    abs_diff = df.with_columns(
        (pl.col("rv_ann") - pl.col("iv_sq_ann")).abs().alias("abs_rv_iv_diff")
    )
    by_dte_diff = abs_diff.group_by("tau_days").agg([
        pl.col("abs_rv_iv_diff").mean().alias("mean_abs_diff"),
        pl.len().alias("n"),
    ]).sort("tau_days")
    axes[1].bar([str(d) for d in by_dte_diff["tau_days"].to_list()],
                by_dte_diff["mean_abs_diff"].to_list(), alpha=0.8)
    axes[1].set_xlabel("Days to expiry")
    axes[1].set_ylabel("Mean |RV − IV²|")
    axes[1].set_title("|RV − IV²| by days-to-expiry (convergence metric)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "rv_iv_convergence_by_dte.png", dpi=dpi)
    plt.close(fig)
    return by_dte


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "configs" / "base.yaml")
    config_hash = compute_config_hash(config)
    cache = project_root / "outputs" / "cache" / config_hash
    out_dir = project_root / "outputs" / "report" / "iv_rv_convergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    svi = pl.read_parquet(cache / "svi_results.parquet")
    options = pl.read_parquet(project_root / "data" / "processed" / config_hash / "options_clean.parquet")
    feat_path = project_root / "outputs" / "features" / f"features_{config_hash}.parquet"
    features = pl.read_parquet(feat_path) if feat_path.exists() else pl.DataFrame()

    print("=" * 60)
    print("IV surface dynamics & RV-IV convergence experiments")
    print("=" * 60)

    run_iv_surface_dynamics(svi, out_dir)
    print("Saved: iv_surface_over_time.png")

    if not features.is_empty():
        run_rv_iv_by_horizon(features, out_dir)
        print("Saved: rv_iv_ratio_by_horizon.png")

    run_convergence_by_dte(svi, options, out_dir)
    print("Saved: rv_iv_convergence_by_dte.png")

    # Write summary markdown
    summary_path = out_dir / "IV_RV_CONVERGENCE_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write("# IV Surface Dynamics & RV-IV Convergence\n\n")
        f.write("## 1. IV Surface Over Time\n")
        f.write("- ATM total variance (w) and skew by tau bucket.\n")
        f.write("- Chart: `iv_surface_over_time.png`\n\n")
        f.write("## 2. RV/IV² by Horizon\n")
        f.write("- Realization ratio (RV/IV²) by horizon; ratio < 1 ⇒ IV rich vs RV.\n")
        f.write("- Chart: `rv_iv_ratio_by_horizon.png`\n\n")
        f.write("## 3. Convergence by Days-to-Expiry\n")
        f.write("- As options approach expiry, RV converges to IV²; ratio → 1.\n")
        f.write("- Chart: `rv_iv_convergence_by_dte.png`\n")
    print(f"Saved: {summary_path}")
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
