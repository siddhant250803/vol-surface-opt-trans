"""W1, W2 quantile-based 1D Wasserstein."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _quantile_from_density(k: np.ndarray, q: np.ndarray, u: float) -> float:
    """Inverse CDF at level u from density (k, q)."""
    if len(k) < 2:
        return float(k[0]) if len(k) == 1 else np.nan
    dx = np.diff(k)
    cdf = np.concatenate([[0], np.cumsum(q[:-1] * dx)])
    total = cdf[-1] + q[-1] * dx[-1] if len(dx) > 0 else cdf[-1]
    if total <= 0:
        return float(k[0])
    cdf = cdf / total
    return float(np.interp(u, cdf, k))


def wasserstein_1(k: np.ndarray, q1: np.ndarray, q2: np.ndarray, n_points: int = 100) -> float:
    """W1 = integral |F1^{-1}(u) - F2^{-1}(u)| du."""
    u = np.linspace(0.01, 0.99, n_points)
    q1_inv = np.array([_quantile_from_density(k, q1, ui) for ui in u])
    q2_inv = np.array([_quantile_from_density(k, q2, ui) for ui in u])
    return float(np.mean(np.abs(q1_inv - q2_inv)))


def wasserstein_2(k: np.ndarray, q1: np.ndarray, q2: np.ndarray, n_points: int = 100) -> float:
    """W2 = (integral (F1^{-1}(u) - F2^{-1}(u))^2 du)^(1/2)."""
    u = np.linspace(0.01, 0.99, n_points)
    q1_inv = np.array([_quantile_from_density(k, q1, ui) for ui in u])
    q2_inv = np.array([_quantile_from_density(k, q2, ui) for ui in u])
    return float(np.sqrt(np.mean((q1_inv - q2_inv) ** 2)))


def compute_all_distances(
    q_densities: pl.DataFrame,
    p_densities: pl.DataFrame,
    config: dict[str, Any],
    q_constant_maturity: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute W1, W2, quantile gaps, tail mass, moments between Q and P, and Q and Q_prev.

    When q_constant_maturity is provided, Q_prev = Q at constant maturity (t-1).
    Otherwise Q_prev = same option series from previous date (rolling maturity).
    """
    from src.distances.quantile_gaps import quantile_gap
    from src.distances.moments import tail_mass_gap, moment_diff

    dist_cfg = config.get("distances", {})
    tail_levels = dist_cfg.get("tail_mass_levels", [0.01, 0.05, 0.95, 0.99])
    use_constant_maturity = dist_cfg.get("use_constant_maturity_q_prev", False) and q_constant_maturity is not None

    q_sorted = q_densities.sort(["date", "tau_bucket"])
    p_by_date_tau = p_densities.group_by(["date", "tau_bucket"]).first()

    dates_sorted = q_sorted["date"].unique().sort().to_list()
    q_by_date_tau: dict[Any, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for i in range(len(q_sorted)):
        row = q_sorted.row(i, named=True)
        dt = row["date"]
        tb = row["tau_bucket"]
        if dt not in q_by_date_tau:
            q_by_date_tau[dt] = {}
        q_by_date_tau[dt][tb] = (np.array(row["k_grid"]), np.array(row["q_density"]))

    q_prev_by_date_tau: dict[Any, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    if use_constant_maturity and q_constant_maturity is not None:
        for i in range(len(q_constant_maturity)):
            row = q_constant_maturity.row(i, named=True)
            dt = row["date"]
            tb = row["tau_bucket"]
            if dt not in q_prev_by_date_tau:
                q_prev_by_date_tau[dt] = {}
            q_prev_by_date_tau[dt][tb] = (np.array(row["k_grid"]), np.array(row["q_density"]))

    rows = []
    for i in range(len(q_sorted)):
        row = q_sorted.row(i, named=True)
        date_val = row["date"]
        tau_bucket = row["tau_bucket"]
        k = np.array(row["k_grid"])
        q = np.array(row["q_density"])

        p_match = p_by_date_tau.filter(
            (pl.col("date") == date_val) & (pl.col("tau_bucket") == tau_bucket)
        )
        if p_match.is_empty():
            continue
        p_row = p_match.row(0, named=True)
        p = np.array(p_row["p_density"])
        k_p = np.array(p_row["k_grid"])

        if len(p) != len(k_p):
            continue
        # Interpolate P to Q's k-grid (P may use bin_centers, Q uses linspace)
        if len(k_p) > 1 and (not np.allclose(k_p, k) or len(k_p) != len(k)):
            p = np.interp(k, k_p, p)
            trapz = np.trapezoid(p, k)
            if trapz > 1e-10:
                p = p / trapz
        k_use = k

        w1_qp = wasserstein_1(k_use, q, p)
        w2_qp = wasserstein_2(k_use, q, p)
        qg1 = quantile_gap(k_use, q, p, 0.01)
        qg5 = quantile_gap(k_use, q, p, 0.05)
        qg50 = quantile_gap(k_use, q, p, 0.5)
        qg95 = quantile_gap(k_use, q, p, 0.95)
        qg99 = quantile_gap(k_use, q, p, 0.99)
        tmg_l, tmg_r = tail_mass_gap(k_use, q, p, tail_levels)
        md1, md2 = moment_diff(k_use, q, p)

        w1_qq, w2_qq = np.nan, np.nan
        idx = dates_sorted.index(date_val) if date_val in dates_sorted else -1
        if idx > 0:
            prev_date = dates_sorted[idx - 1]
            q_prev_source = q_prev_by_date_tau if use_constant_maturity else q_by_date_tau
            if prev_date in q_prev_source and tau_bucket in q_prev_source[prev_date]:
                k_prev, q_prev = q_prev_source[prev_date][tau_bucket]
                k_prev = np.asarray(k_prev)
                q_prev = np.asarray(q_prev)
                if len(k_prev) > 1 and len(q_prev) == len(k_prev):
                    q_prev_on_k = np.interp(k_use, k_prev, q_prev)
                    trapz = np.trapezoid(q_prev_on_k, k_use)
                    if trapz > 1e-10:
                        q_prev_on_k = q_prev_on_k / trapz
                    w1_qq = wasserstein_1(k_use, q, q_prev_on_k)
                    w2_qq = wasserstein_2(k_use, q, q_prev_on_k)

        rows.append({
            "date": date_val,
            "exdate": row["exdate"],
            "tau_bucket": tau_bucket,
            "target_tau_days": row["target_tau_days"],
            "forward": row["forward"],
            "spx_price": row["spx_price"],
            "w1_q_p": w1_qp,
            "w2_q_p": w2_qp,
            "w1_q_q_prev": w1_qq,
            "w2_q_q_prev": w2_qq,
            "quantile_gap_1": qg1,
            "quantile_gap_5": qg5,
            "quantile_gap_50": qg50,
            "quantile_gap_95": qg95,
            "quantile_gap_99": qg99,
            "tail_mass_gap_left": tmg_l,
            "tail_mass_gap_right": tmg_r,
            "moment_diff_1": md1,
            "moment_diff_2": md2,
            "fit_status": row["fit_status"],
            "q_integral": row["q_integral"],
            "invalid_density": row["invalid_density"],
        })

    return pl.DataFrame(rows)
