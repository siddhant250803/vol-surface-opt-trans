"""Golden test: hard assertions and dedicated artifacts for 3 dates x 2 tau."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.surface.svi_fit import _svi_total_var, _iv_from_total_var

logger = logging.getLogger("pipeline.golden")


class GoldenAssertionError(Exception):
    """Raised when a golden assertion fails."""

    pass


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise GoldenAssertionError(msg)


def run_golden_assertions(
    options_clean: pl.DataFrame,
    tau_mapped: pl.DataFrame,
    svi_results: pl.DataFrame,
    call_prices: pl.DataFrame,
    q_densities: pl.DataFrame,
    p_densities: pl.DataFrame,
    distances: pl.DataFrame,
    features: pl.DataFrame,
    rates: pl.DataFrame,
    config: dict[str, Any],
) -> pl.DataFrame:
    """Run hard assertions. Mark invalid slices. Returns features with golden_valid column."""
    gc = config.get("golden", {})
    ac = gc.get("assertions", {})
    fit_cfg = config.get("fit", {})
    filters_cfg = config.get("filters", {})

    min_strikes = ac.get("min_strikes", filters_cfg.get("min_strikes_per_expiry", 20))
    r_min = ac.get("r_annual_min", -0.02)
    r_max = ac.get("r_annual_max", 0.15)
    iv_rmse_max = ac.get("iv_rmse_max", fit_cfg.get("iv_rmse_max", 0.05))
    iv_std_min = ac.get("fitted_iv_std_min", 0.001)
    q_int_min = ac.get("q_integral_min", 0.995)
    q_int_max = ac.get("q_integral_max", 1.005)
    neg_mass_max = ac.get("neg_mass_max", 0.01)
    clip_frac_max = ac.get("clip_frac_max", 0.01)
    w1_qq_min = float(ac.get("w1_qq_min_nonzero", 1e-6))
    rv_nonnull_frac = ac.get("rv_nonnull_min_frac", 0.5)

    failures = []
    valid_mask = []

    for i in range(len(features)):
        row = features.row(i, named=True)
        date_val = row["date"]
        tau_bucket = row["tau_bucket"]
        slice_failures = []

        if "iv_rmse_vega_weighted" in row and row["iv_rmse_vega_weighted"] is not None:
            if row["iv_rmse_vega_weighted"] > iv_rmse_max:
                slice_failures.append(f"IV_RMSE={row['iv_rmse_vega_weighted']:.4f} > {iv_rmse_max}")

        if "q_integral" in row and row["q_integral"] is not None:
            qi = row["q_integral"]
            if qi < q_int_min or qi > q_int_max:
                slice_failures.append(f"q_integral={qi:.4f} not in [{q_int_min},{q_int_max}]")

        if "q_neg_clip_frac" in row and row["q_neg_clip_frac"] is not None:
            if row["q_neg_clip_frac"] > neg_mass_max:
                slice_failures.append(f"neg_mass={row['q_neg_clip_frac']:.4f} > {neg_mass_max}")

        valid_mask.append(len(slice_failures) == 0)
        if slice_failures:
            failures.append((str(date_val), tau_bucket, slice_failures))

    for (d, tb, msgs) in failures:
        logger.warning("Golden FAIL (%s %s): %s", d, tb, "; ".join(msgs))

    features = features.with_columns(pl.Series("golden_valid", valid_mask))

    w1_qq = features["w1_q_q_prev"].drop_nulls()
    if len(w1_qq) > 0 and w1_qq_min > 0:
        max_w1 = float(w1_qq.max())
        if max_w1 < w1_qq_min:
            raise GoldenAssertionError(
                f"W1(Q,Q_prev) max={max_w1:.2e} < {w1_qq_min}; OT/CDF likely broken"
            )

    rv_cols = [c for c in features.columns if c.startswith("realized_var_")]
    if rv_cols:
        rv = features[rv_cols[0]]
        nonnull = rv.drop_nulls()
        if len(nonnull) < len(features) * rv_nonnull_frac:
            raise GoldenAssertionError(
                f"RV non-null {len(nonnull)}/{len(features)} < {rv_nonnull_frac*100}%"
            )
        if (nonnull < 0).any():
            raise GoldenAssertionError("Realized variance has negative values")

    w2_qp = features.filter(pl.col("date").cast(str).str.contains("2020-03"))["w2_q_p"].drop_nulls()
    w2_calm = features.filter(pl.col("date").cast(str).str.contains("2017-06"))["w2_q_p"].drop_nulls()
    if len(w2_qp) > 0 and len(w2_calm) > 0:
        diff = abs(float(w2_qp.mean()) - float(w2_calm.mean()))
        if diff < 1e-10:
            raise GoldenAssertionError(
                "W2(Q,P) same on 2020 stress vs 2017 calm; distributions not differentiated"
            )

    return features


def generate_golden_artifacts(
    tau_mapped: pl.DataFrame,
    svi_results: pl.DataFrame,
    call_prices: pl.DataFrame,
    q_densities: pl.DataFrame,
    p_densities: pl.DataFrame,
    distances: pl.DataFrame,
    features: pl.DataFrame,
    options_clean: pl.DataFrame,
    config: dict[str, Any],
    report_dir: Path,
    config_hash: str,
) -> None:
    """Generate golden charts and tables under outputs/report/{config_hash}/golden/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    golden_dir = report_dir / config_hash / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    k_cfg = config.get("density", {}).get("k_grid", {})
    k_grid = np.linspace(k_cfg.get("k_min", -0.5), k_cfg.get("k_max", 0.5), k_cfg.get("n_points", 201))

    for i in range(len(svi_results)):
        row = svi_results.row(i, named=True)
        date_val = row["date"]
        tau_bucket = row["tau_bucket"]
        a, b, rho, m, sigma = row["svi_a"], row["svi_b"], row["svi_rho"], row["svi_m"], row["svi_sigma"]

        sub = tau_mapped.filter(
            (pl.col("date") == date_val) & (pl.col("tau_bucket") == tau_bucket)
        )
        if sub.is_empty():
            continue

        forward = row["forward"]
        tau = row["tau"]
        strike = sub["strike_price"].to_numpy() / 1000.0
        k_mkt = np.log(strike / forward)
        iv_mkt = sub["impl_volatility"].to_numpy()

        w = _svi_total_var(k_grid, a, b, rho, m, sigma)
        iv_fit = _iv_from_total_var(w, tau)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(k_mkt, iv_mkt, label="Market IV", alpha=0.7)
        ax.plot(k_grid, iv_fit, "r-", label="Fitted IV")
        ax.set_xlabel("Log-moneyness k")
        ax.set_ylabel("Implied volatility")
        ax.set_title(f"Market vs Fitted IV: {date_val} {tau_bucket}")
        ax.legend()
        ax.axhline(0, color="k", ls=":", alpha=0.3)
        fig.tight_layout()
        fig.savefig(golden_dir / f"iv_smile_{date_val}_{tau_bucket}.png", dpi=150)
        plt.close(fig)

    for i in range(len(call_prices)):
        row = call_prices.row(i, named=True)
        date_val = row["date"]
        tau_bucket = row["tau_bucket"]
        forward = row["forward"]
        k_grid = np.array(row["k_grid"])
        c_fit = np.array(row["call_prices"])
        sub = tau_mapped.filter(
            (pl.col("date") == date_val) & (pl.col("tau_bucket") == tau_bucket) & (pl.col("cp_flag") == "C")
        )
        if sub.is_empty():
            continue
        strike = sub["strike_price"].to_numpy() / 1000.0
        k_mkt = np.log(strike / forward)
        mid = (sub["best_bid"].to_numpy() + sub["best_offer"].to_numpy()) / 2.0
        c_fit_at_k = np.interp(k_mkt, k_grid, c_fit)
        c_fit_index_pts = float(forward) * c_fit_at_k
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(k_mkt, mid, label="Market mid", alpha=0.7)
        ax.scatter(k_mkt, c_fit_index_pts, label="Fitted", alpha=0.7, marker="x")
        ax.set_xlabel("Log-moneyness k")
        ax.set_ylabel("Call price (index pts)")
        ax.set_title(f"Market vs Fitted Call Prices: {date_val} {tau_bucket}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(golden_dir / f"call_prices_{date_val}_{tau_bucket}.png", dpi=150)
        plt.close(fig)

    for i in range(len(q_densities)):
        row = q_densities.row(i, named=True)
        date_val = row["date"]
        tau_bucket = row["tau_bucket"]
        k = np.array(row["k_grid"])
        q = np.array(row["q_density"])
        cdf = np.array(row["q_cdf"]) if "q_cdf" in row and row["q_cdf"] is not None else np.concatenate([[0], np.cumsum(q[:-1] * np.diff(k))]) if len(np.diff(k)) > 0 else np.zeros_like(k)
        if "q_cdf" not in row or row["q_cdf"] is None:
            total = cdf[-1] + (q[-1] * np.diff(k)[-1] if len(np.diff(k)) > 0 else 0)
            if total > 0:
                cdf = cdf / total

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(k, q)
        axes[0].set_xlabel("x = ln(S_T/F)")
        axes[0].set_ylabel("PDF q(x)")
        axes[0].set_title(f"Q density: {date_val} {tau_bucket}")
        axes[1].plot(k, cdf)
        axes[1].set_xlabel("x = ln(S_T/F)")
        axes[1].set_ylabel("CDF")
        axes[1].set_title(f"Q CDF: {date_val} {tau_bucket}")
        fig.tight_layout()
        fig.savefig(golden_dir / f"q_density_{date_val}_{tau_bucket}.png", dpi=150)
        plt.close(fig)

    for tau_bucket in ["7D", "30D"]:
        q_sub = q_densities.filter(pl.col("tau_bucket") == tau_bucket)
        if q_sub.is_empty():
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(len(q_sub)):
            row = q_sub.row(i, named=True)
            k = np.array(row["k_grid"])
            q = np.array(row["q_density"])
            ax.plot(k, q, label=str(row["date"]), alpha=0.8)
        ax.set_xlabel("x = ln(S_T/F)")
        ax.set_ylabel("PDF")
        ax.set_title(f"Q overlay {tau_bucket}: 2017 vs 2020 vs 2022")
        ax.legend()
        fig.tight_layout()
        fig.savefig(golden_dir / f"q_overlay_{tau_bucket}.png", dpi=150)
        plt.close(fig)

    # Key overlay: 2017-06-02, 2020-03-06, 2022-10-07 7D (must show 2020 fatter left tail)
    anchor_dates = ["2017-06-02", "2020-03-06", "2022-10-07"]
    q_anchor = q_densities.filter(
        (pl.col("tau_bucket") == "7D") & (pl.col("date").cast(str).is_in(anchor_dates))
    )
    if not q_anchor.is_empty():
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"2017-06-02": "green", "2020-03-06": "red", "2022-10-07": "blue"}
        for i in range(len(q_anchor)):
            row = q_anchor.row(i, named=True)
            k = np.array(row["k_grid"])
            q = np.array(row["q_density"])
            d = str(row["date"])
            ax.plot(k, q, label=d, color=colors.get(d, "gray"), alpha=0.8, lw=2)
        ax.set_xlabel("x = ln(S_T/F)")
        ax.set_ylabel("PDF q(x)")
        ax.set_title("Q overlay: 2017 (calm) vs 2020 (stress) vs 2022 (recent)")
        ax.legend()
        ax.axvline(0, color="k", ls=":", alpha=0.3)
        fig.tight_layout()
        fig.savefig(golden_dir / "q_overlay_anchor_dates.png", dpi=150)
        plt.close(fig)

    # Convexity diagnostic: ∂²C/∂K² for one slice (2020-03-06 7D)
    conv_row = None
    for i in range(len(call_prices)):
        row = call_prices.row(i, named=True)
        if str(row["date"]) == "2020-03-06" and row["tau_bucket"] == "7D":
            conv_row = row
            break
    if conv_row is not None:
        k = np.array(conv_row["k_grid"])
        c = np.array(conv_row["call_prices"])
        F = float(conv_row["forward"])
        dk = np.diff(k)
        dx = float(np.median(dk)) if len(dk) > 0 else 0.01
        d2c_dk2 = np.zeros_like(k)
        d2c_dk2[1:-1] = (c[2:] - 2 * c[1:-1] + c[:-2]) / (dx**2)
        d2c_dk2[0], d2c_dk2[-1] = d2c_dk2[1], d2c_dk2[-2]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(k, d2c_dk2, "b-")
        ax.axhline(0, color="k", ls=":", alpha=0.3)
        ax.set_xlabel("Log-moneyness k")
        ax.set_ylabel("∂²C/∂K² (convexity)")
        ax.set_title("Convexity (2020-03-06 7D): must be > 0 for no arbitrage")
        fig.tight_layout()
        fig.savefig(golden_dir / "convexity_d2C_dK2.png", dpi=150)
        plt.close(fig)

    # Call price curve diagnostic: fitted C vs strike, straight-line overlay, ATM zoom
    if conv_row is not None:
        k = np.array(conv_row["k_grid"])
        c_norm = np.array(conv_row["call_prices"])
        F = float(conv_row["forward"])
        strike = F * np.exp(k)
        c_dollars = F * c_norm
        line_slope = (c_dollars[-1] - c_dollars[0]) / (strike[-1] - strike[0]) if strike[-1] != strike[0] else 0
        line_intercept = c_dollars[0] - line_slope * strike[0]
        line_vals = line_slope * strike + line_intercept

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(strike, c_dollars, "b-", label="Fitted C(K)")
        axes[0].plot(strike, line_vals, "r--", alpha=0.7, label="Straight line (first–last)")
        axes[0].set_xlabel("Strike K")
        axes[0].set_ylabel("Call price C")
        axes[0].set_title("Call price vs strike (2020-03-06 7D)")
        axes[0].legend()
        axes[0].axhline(0, color="k", ls=":", alpha=0.3)

        atm_mask = np.abs(k) < 0.1
        if atm_mask.sum() > 2:
            k_atm = k[atm_mask]
            c_atm = c_dollars[atm_mask]
            s_atm = strike[atm_mask]
            line_atm = line_slope * s_atm + line_intercept
            axes[1].plot(s_atm, c_atm, "b-", label="Fitted C(K)")
            axes[1].plot(s_atm, line_atm, "r--", alpha=0.7, label="Straight line")
            axes[1].set_xlabel("Strike K")
            axes[1].set_ylabel("Call price C")
            axes[1].set_title("ATM zoom (|k| < 0.1)")
            axes[1].legend()
        else:
            axes[1].plot(strike, c_dollars, "b-")
            axes[1].plot(strike, line_vals, "r--", alpha=0.7)
            axes[1].set_title("ATM zoom (full range)")
        fig.tight_layout()
        fig.savefig(golden_dir / "call_price_curve_diagnostic.png", dpi=150)
        plt.close(fig)

    dist_cols = [c for c in ["date", "tau_bucket", "w1_q_p", "w2_q_p", "w1_q_q_prev", "w2_q_q_prev", "svi_param_dist_prev", "quantile_gap_1", "quantile_gap_5", "quantile_gap_50", "quantile_gap_95", "quantile_gap_99"] if c in features.columns]
    if dist_cols:
        features.select(dist_cols).write_csv(golden_dir / "distance_table.csv")

    rv_cols = [c for c in features.columns if "realized_var" in c]
    with open(golden_dir / "rv_join_diagnostic.txt", "w") as f:
        for c in rv_cols:
            nonnull = features[c].drop_nulls().len()
            f.write(f"{c}: nonnull={nonnull}/{len(features)}\n")

    for i in range(min(6, len(p_densities))):
        row = p_densities.row(i, named=True)
        k = np.array(row["k_grid"])
        p = np.array(row["p_density"])
        if len(k) > 1:
            mean_p = float(np.trapezoid(k * p, k))
            var_p = float(np.trapezoid(((k - mean_p) ** 2) * p, k))
            skew_p = float(np.trapezoid(((k - mean_p) ** 3) * p, k) / (var_p**1.5 + 1e-12))
            kurt_p = float(np.trapezoid(((k - mean_p) ** 4) * p, k) / (var_p**2 + 1e-12) - 3)
        else:
            mean_p = var_p = skew_p = kurt_p = np.nan
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k, p, "b-", label="P density")
        ax.fill_between(k, p, alpha=0.3)
        stats_txt = f"mean={mean_p:.4f} var={var_p:.4f} skew={skew_p:.2f} kurt={kurt_p:.2f}"
        ax.text(0.02, 0.98, stats_txt, transform=ax.transAxes, fontsize=9, verticalalignment="top")
        ax.set_xlabel("x = ln(S_T/F)")
        ax.set_ylabel("Density")
        ax.set_title(f"P distribution: {row['date']} {row['tau_bucket']}")
        ax.axhline(0, color="k", ls=":", alpha=0.3)
        fig.tight_layout()
        fig.savefig(golden_dir / f"p_dist_{row['date']}_{row['tau_bucket']}.png", dpi=150)
        plt.close(fig)

    with open(golden_dir / "q_density_report.txt", "w") as f:
        f.write("date\ttau_bucket\tintegral\tneg_clip_frac\tmass_outside\n")
        for i in range(len(q_densities)):
            row = q_densities.row(i, named=True)
            f.write(f"{row['date']}\t{row['tau_bucket']}\t{row.get('q_integral', '')}\t{row.get('q_neg_clip_frac', '')}\t{row.get('q_mass_outside_grid', '')}\n")

    with open(golden_dir / "diagnostics.txt", "w") as f:
        f.write(f"Golden run config_hash: {config_hash}\n")
        f.write(f"SVI slices: {len(svi_results)}\n")
        f.write(f"Q slices: {len(q_densities)}\n")
        f.write(f"Distance slices: {len(distances)}\n")
