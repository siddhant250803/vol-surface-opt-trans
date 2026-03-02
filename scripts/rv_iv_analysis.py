"""RV-IV decile table, cost-adjusted Sharpe, subperiod stability.

TRADABLE DECILE: Sort by predictor (W1) known at entry, then compute forward RV−IV².
Ex-post sort by outcome is tautological.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from pipeline.config import load_config, compute_config_hash


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "configs" / "base.yaml")
    config_hash = compute_config_hash(config)
    feat = pl.read_parquet(project_root / "outputs" / "features" / f"features_{config_hash}.parquet")

    # Convention 2: annualized variance. Same horizon (7D) for RV and IV.
    tau_7d = 7 / 365
    rv_col = "realized_var_7d_ann" if "realized_var_7d_ann" in feat.columns else "realized_var_7d"
    if rv_col == "realized_var_7d":
        rv_ann_expr = pl.col("realized_var_7d") * (365 / 7)
    else:
        rv_ann_expr = pl.col("realized_var_7d_ann")

    feat = feat.with_columns([
        (pl.col("atm_iv") / tau_7d).alias("iv_sq_ann"),
        rv_ann_expr.alias("rv_ann"),
    ])
    feat = feat.with_columns((pl.col("rv_ann") - pl.col("iv_sq_ann")).alias("rv_iv_diff"))

    feat = feat.filter(
        pl.col("rv_iv_diff").is_not_null()
        & pl.col("rv_iv_diff").is_finite()
        & pl.col("iv_sq_ann").gt(0)
        & pl.col("iv_sq_ann").lt(2)  # sanity: IV² < 2 (vol < 141%)
    )
    if feat.is_empty():
        print("No rows after filter. Check rv_ann, iv_sq_ann.")
        return

    # --- TRADABLE DECILE: sort by W1 (predictor known at entry), not by outcome ---
    predictor = "w1_q_q_prev" if "w1_q_q_prev" in feat.columns else "w1_q_p"
    if predictor not in feat.columns:
        print("No W1 predictor; falling back to ex-post decile (not tradable).")
        sort_col = "rv_iv_diff"
    else:
        sort_col = predictor
        print("Deciles formed by predictor (tradable):", predictor)

    feat = feat.filter(pl.col(sort_col).is_not_null())
    feat = feat.with_columns(
        pl.col(sort_col).qcut(10, labels=[f"D{i}" for i in range(1, 11)]).alias("decile")
    )

    # --- 1. DECILE TABLE ---
    decile_table = feat.group_by("decile").agg([
        pl.col("rv_iv_diff").mean().alias("mean_rv_iv_diff"),
        pl.col("rv_ann").mean().alias("mean_rv_ann"),
        pl.col("iv_sq_ann").mean().alias("mean_iv_sq"),
        (pl.col("iv_sq_ann").sqrt() * 100).mean().alias("mean_iv_pct"),
        pl.len().alias("n"),
    ]).sort("decile")

    print("=" * 75)
    print("DECILE TABLE: mean(RV − IV²) by decile of", sort_col)
    print("=" * 75)
    print("RV_ann, IV²_ann: annualized variance. Same 7D horizon.")
    print("D1 = low predictor, D10 = high predictor. Forward outcome = RV − IV².")
    print(decile_table)
    print()

    # --- 2. COST-ADJUSTED SHARPE ---
    feat = feat.with_columns(
        pl.when(pl.col("decile") == "D10")
        .then(pl.col("rv_iv_diff"))
        .when(pl.col("decile") == "D1")
        .then(-pl.col("rv_iv_diff"))
        .otherwise(0.0)
        .alias("strat_return")
    )

    feat = feat.sort("date")
    rets = feat["strat_return"].to_numpy()
    n_trades = (feat["decile"].is_in(["D1", "D10"])).sum()

    ann_factor = np.sqrt(52)
    mean_ret = float(np.nanmean(rets))
    std_ret = float(np.nanstd(rets))
    sharpe_raw = mean_ret / std_ret * ann_factor if std_ret > 1e-10 else 0.0

    cost_bps = 5
    cost_per_trade = cost_bps / 10000
    cost_per_period = (n_trades * cost_per_trade) / len(feat) if len(feat) > 0 else 0
    rets_net = rets - cost_per_period
    sharpe_net = float(np.nanmean(rets_net)) / float(np.nanstd(rets_net)) * ann_factor if np.nanstd(rets_net) > 1e-10 else 0.0

    print("=" * 75)
    print("COST-ADJUSTED SHARPE")
    print("=" * 75)
    print(f"Strategy: Long D10 (high {sort_col}), Short D1 (low {sort_col})")
    print(f"Raw Sharpe (annualized):     {sharpe_raw:.3f}")
    print(f"Cost: {cost_bps} bps round-trip per trade")
    print(f"Cost-adjusted Sharpe:        {sharpe_net:.3f}")
    print(f"N in D1 or D10: {n_trades} / {len(feat)}")
    print()

    # --- 3. SUBPERIOD STABILITY ---
    feat = feat.with_columns(pl.col("date").dt.year().alias("year"))

    subperiods = [
        ("2013-2015", (2013, 2015)),
        ("2016-2018", (2016, 2018)),
        ("2019-2021", (2019, 2021)),
        ("2022-2023", (2022, 2023)),
    ]

    print("=" * 75)
    print("SUBPERIOD STABILITY")
    print("=" * 75)

    rows = []
    for label, (y0, y1) in subperiods:
        sub = feat.filter((pl.col("year") >= y0) & (pl.col("year") <= y1))
        if len(sub) < 10:
            continue
        rets_sub = sub["strat_return"].to_numpy()
        mean_s = float(np.nanmean(rets_sub))
        std_s = float(np.nanstd(rets_sub))
        sharpe_s = mean_s / std_s * ann_factor if std_s > 1e-10 else 0.0
        rv_iv = sub["rv_iv_diff"].mean()
        rows.append({"period": label, "n": len(sub), "mean_ret": mean_s, "std_ret": std_s, "sharpe": sharpe_s, "mean_rv_iv": rv_iv})

    tbl = pl.DataFrame(rows)
    print(tbl)
    print()

    print("Decile spread (D10 - D1 mean rv_iv_diff) by subperiod:")
    for label, (y0, y1) in subperiods:
        sub = feat.filter((pl.col("year") >= y0) & (pl.col("year") <= y1))
        if len(sub) < 10:
            continue
        d10_df = sub.filter(pl.col("decile") == "D10")
        d1_df = sub.filter(pl.col("decile") == "D1")
        d10_val = d10_df["rv_iv_diff"].mean()
        d1_val = d1_df["rv_iv_diff"].mean()
        d10 = float(d10_val) if d10_val is not None and len(d10_df) > 0 else None
        d1 = float(d1_val) if d1_val is not None and len(d1_df) > 0 else None
        if d10 is not None and d1 is not None:
            print(f"  {label}: {d10 - d1:.4f}  (D10={d10:.4f}, D1={d1:.4f})")


if __name__ == "__main__":
    main()
