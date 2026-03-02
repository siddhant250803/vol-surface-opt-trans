"""Map (date, target_tau) to nearest expiry."""

from __future__ import annotations

from typing import Any

import polars as pl


def map_tau_buckets(df: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """For each (date, exdate), assign target tau bucket if within tolerance."""
    buckets = config.get("tau_buckets", [])
    if not buckets:
        return df.with_columns(
            pl.lit(None).alias("target_tau_days"),
            pl.lit(None).alias("tau_bucket"),
        )

    out = df.with_columns(
        (pl.col("exdate") - pl.col("date")).dt.total_days().alias("tau_days")
    )

    rows = []
    for bucket in buckets:
        target = bucket["target_days"]
        tol = bucket.get("tolerance_days", 5)
        label = bucket.get("label", str(target))

        mask = (pl.col("tau_days") >= target - tol) & (pl.col("tau_days") <= target + tol)
        sub = out.filter(mask).with_columns(
            pl.lit(target).alias("target_tau_days"),
            pl.lit(label).alias("tau_bucket"),
        )
        rows.append(sub)

    if not rows:
        return out.with_columns(
            pl.lit(None).alias("target_tau_days"),
            pl.lit(None).alias("tau_bucket"),
        )

    return pl.concat(rows)
