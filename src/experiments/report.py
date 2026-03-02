"""Generate factor charts for the report (W1, W2, ATM IV, skew)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def generate_all_charts(features: pl.DataFrame, config: dict[str, Any], report_dir: Path) -> None:
    """Generate factor charts (W1, W2, ATM IV, skew time series)."""
    plot_cfg = config.get("plotting", config.get("configs", {}).get("plotting", {}))
    if isinstance(plot_cfg, list):
        plot_cfg = {}
    dpi = plot_cfg.get("dpi", 150)
    fmt = plot_cfg.get("format", "png")

    (report_dir / "factor").mkdir(parents=True, exist_ok=True)
    if features.is_empty():
        return

    if "atm_iv" in features.columns:
        _chart_atm_iv_ts(features, report_dir / "factor", dpi, fmt)
    if "skew" in features.columns:
        _chart_skew_ts(features, report_dir / "factor", dpi, fmt)
    if "w1_q_q_prev" in features.columns:
        _chart_w1_q_vs_q_ts(features, report_dir / "factor", dpi, fmt)
    if "svi_param_dist_prev" in features.columns:
        _chart_svi_param_dist_ts(features, report_dir / "factor", dpi, fmt)
    if "w1_q_p" in features.columns:
        _chart_w2_q_vs_p_ts(features, report_dir / "factor", dpi, fmt)


def _chart_atm_iv_ts(df: pl.DataFrame, out: Path, dpi: int, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    sub = df.filter(pl.col("tau_bucket") == df["tau_bucket"].mode().first()).sort("date")
    if not sub.is_empty():
        ax.plot(sub["date"].to_list(), sub["atm_iv"].to_list())
    ax.set_xlabel("Date")
    ax.set_ylabel("ATM IV")
    ax.set_title("ATM IV time series")
    fig.tight_layout()
    fig.savefig(out / ("atm_iv_time_series." + fmt), dpi=dpi)
    plt.close(fig)


def _chart_skew_ts(df: pl.DataFrame, out: Path, dpi: int, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    sub = df.filter(pl.col("tau_bucket") == df["tau_bucket"].mode().first()).sort("date")
    if not sub.is_empty():
        ax.plot(sub["date"].to_list(), sub["skew"].to_list())
    ax.set_xlabel("Date")
    ax.set_ylabel("Skew")
    ax.set_title("Skew time series")
    fig.tight_layout()
    fig.savefig(out / ("skew_time_series." + fmt), dpi=dpi)
    plt.close(fig)


def _chart_svi_param_dist_ts(df: pl.DataFrame, out: Path, dpi: int, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    sub = df.filter(pl.col("tau_bucket").is_not_null()).sort("date")
    sp = sub["svi_param_dist_prev"].drop_nulls()
    if len(sp) > 0:
        dates = sub.filter(pl.col("svi_param_dist_prev").is_not_null())["date"].to_list()
        ax.plot(dates, sp.to_list())
    ax.set_xlabel("Date")
    ax.set_ylabel("SVI param dist (θ_t vs θ_{t-1})")
    ax.set_title("SVI parameter-space distance time series")
    fig.tight_layout()
    fig.savefig(out / ("svi_param_dist_ts." + fmt), dpi=dpi)
    plt.close(fig)


def _chart_w1_q_vs_q_ts(df: pl.DataFrame, out: Path, dpi: int, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    sub = df.filter(pl.col("tau_bucket").is_not_null()).sort("date")
    w1 = sub["w1_q_q_prev"].drop_nulls()
    if len(w1) > 0:
        dates = sub.filter(pl.col("w1_q_q_prev").is_not_null())["date"].to_list()
        ax.plot(dates, w1.to_list())
    ax.set_xlabel("Date")
    ax.set_ylabel("W1(Q, Q_prev)")
    ax.set_title("W1 Q vs Q_prev time series")
    fig.tight_layout()
    fig.savefig(out / ("w1_q_vs_q_ts." + fmt), dpi=dpi)
    plt.close(fig)


def _chart_w2_q_vs_p_ts(df: pl.DataFrame, out: Path, dpi: int, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    sub = df.sort("date")
    w2 = sub["w2_q_p"].drop_nulls()
    if len(w2) > 0:
        dates = sub.filter(pl.col("w2_q_p").is_not_null())["date"].to_list()
        ax.plot(dates[: len(w2)], w2.to_list())
    ax.set_xlabel("Date")
    ax.set_ylabel("W2(Q, P)")
    ax.set_title("W2 Q vs P time series")
    fig.tight_layout()
    fig.savefig(out / ("w2_q_vs_p_ts." + fmt), dpi=dpi)
    plt.close(fig)


