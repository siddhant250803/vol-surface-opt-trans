"""SVI fit, no-arb checks, fit_status, IV_RMSE. IV decimal, w = iv²*T (years).

Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
Quantitative Finance, 14(1), 59–71.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.units import assert_iv_bounds, assert_total_var_bounds
import polars as pl
from scipy.optimize import minimize


def _svi_total_var(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """SVI raw parameterization: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    d = np.sqrt((k - m) ** 2 + sigma**2)
    return a + b * (rho * (k - m) + d)


def _iv_from_total_var(w: np.ndarray, tau: float) -> np.ndarray:
    """Convert total variance to implied vol: sigma = sqrt(w / tau)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        iv = np.sqrt(np.maximum(w / tau, 1e-12))
    return np.where(np.isfinite(iv), iv, np.nan)


def _bs_call_from_iv(k: np.ndarray, iv: np.ndarray, tau: float) -> np.ndarray:
    """Black-Scholes call C/F from log-moneyness k and IV. sigma = iv*sqrt(tau)."""
    from scipy.stats import norm
    sigma = np.maximum(iv * np.sqrt(tau), 1e-8)
    d1 = -k / sigma + 0.5 * sigma
    d2 = d1 - sigma
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)


def _objective_total_var(params: np.ndarray, k: np.ndarray, w_mkt: np.ndarray, vega: np.ndarray) -> float:
    """Minimize (w_mkt - w_fit)^2. No clipping. Fit total variance directly."""
    a, b, rho, m, sigma = params
    w_fit = _svi_total_var(k, a, b, rho, m, sigma)
    err = w_fit - w_mkt
    vw = vega if vega is not None else np.ones_like(k)
    vw = np.maximum(vw, 1e-6)
    return float(np.sqrt(np.average(err**2, weights=vw)))


def _objective(params: np.ndarray, k: np.ndarray, iv_mkt: np.ndarray, vega: np.ndarray, tau: float, use_price_error: bool = False, fit_total_var: bool = True) -> float:
    """Default: fit total variance w(k). Set fit_total_var=False for legacy IV fit."""
    if fit_total_var:
        w_mkt = (iv_mkt ** 2) * tau
        return _objective_total_var(params, k, w_mkt, vega)
    a, b, rho, m, sigma = params
    w = _svi_total_var(k, a, b, rho, m, sigma)
    iv_fit = _iv_from_total_var(w, tau)
    vw = vega if vega is not None else np.ones_like(k)
    vw = np.maximum(vw, 1e-6)
    if use_price_error:
        c_mkt = _bs_call_from_iv(k, iv_mkt, tau)
        c_fit = _bs_call_from_iv(k, iv_fit, tau)
        err = c_fit - c_mkt
        return np.sqrt(np.average(err**2, weights=vw))
    err = iv_fit - iv_mkt
    return np.sqrt(np.average(err**2, weights=vw))


def _no_arb_check(a: float, b: float, rho: float, m: float, sigma: float) -> tuple[bool, str]:
    """No-arbitrage: b>0, sigma>0, butterfly b*(1+|rho|)<=4/sigma^2."""
    if b <= 0 or sigma <= 0:
        return False, "b or sigma non-positive"
    if b * (1 + np.abs(rho)) > 4 / (sigma**2):
        return False, "butterfly violation"
    return True, ""


def fit_svi_slice(
    k: np.ndarray,
    iv: np.ndarray,
    vega: np.ndarray | None,
    tau: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit SVI to one slice. Returns params, iv_rmse, fit_status, warnings."""
    svi_cfg = config.get("svi", {})
    fit_total_var = svi_cfg.get("fit_total_variance", True)
    bounds_cfg = svi_cfg.get("param_bounds", {})

    w_mkt_mean = float(np.mean((iv ** 2) * tau))
    w_scale = max(w_mkt_mean, 1e-6)
    a_init = w_mkt_mean * 0.5
    b_init = w_mkt_mean * 2.0

    bounds = [
        (bounds_cfg.get("a", [1e-8, 0.5])[0], bounds_cfg.get("a", [1e-8, 0.5])[1]),
        (bounds_cfg.get("b", [1e-8, 0.5])[0], bounds_cfg.get("b", [1e-8, 0.5])[1]),
        (bounds_cfg.get("rho", [-0.99, 0.99])[0], bounds_cfg.get("rho", [-0.99, 0.99])[1]),
        (bounds_cfg.get("m", [-0.5, 0.5])[0], bounds_cfg.get("m", [-0.5, 0.5])[1]),
        (bounds_cfg.get("sigma", [1e-6, 0.5])[0], bounds_cfg.get("sigma", [1e-6, 0.5])[1]),
    ]

    x0 = np.array([a_init, b_init, -0.3, 0.0, 0.15])
    vega_arr = vega if vega is not None else np.ones_like(k)
    use_price_error = svi_cfg.get("vega_weighted_price_error", False)

    try:
        res = minimize(
            lambda p: _objective(p, k, iv, vega_arr, tau, use_price_error, fit_total_var),
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": svi_cfg.get("max_iter", 1000), "ftol": 1e-12},
        )
        a, b, rho, m, sigma = res.x
        w_fit = _svi_total_var(k, a, b, rho, m, sigma)
        iv_fit = _iv_from_total_var(w_fit, tau)
        iv_rmse = float(np.sqrt(np.average((iv_fit - iv) ** 2, weights=vega_arr)))
        if svi_cfg.get("assert_units", False):
            assert_iv_bounds(iv_fit, "IV_fit")
            assert_total_var_bounds(w_fit, "w_fit")
        no_arb, msg = _no_arb_check(a, b, rho, m, sigma) if svi_cfg.get("no_arb_check", True) else (True, "")
        fit_status = "ok" if res.success and no_arb else "fail"
        warnings = [] if res.success else [f"opt: {res.message}"]
        if not no_arb:
            warnings.append(msg)
        return {
            "svi_a": a,
            "svi_b": b,
            "svi_rho": rho,
            "svi_m": m,
            "svi_sigma": sigma,
            "iv_rmse_vega_weighted": iv_rmse,
            "fit_status": fit_status,
            "n_strikes": len(k),
            "no_arb_violations": 0 if no_arb else 1,
            "warnings": "; ".join(warnings) if warnings else "",
        }
    except Exception as e:
        return {
            "svi_a": np.nan,
            "svi_b": np.nan,
            "svi_rho": np.nan,
            "svi_m": np.nan,
            "svi_sigma": np.nan,
            "iv_rmse_vega_weighted": np.nan,
            "fit_status": "fail",
            "n_strikes": len(k),
            "no_arb_violations": 1,
            "warnings": str(e),
        }


def fit_svi_surface(tau_mapped: pl.DataFrame, rates: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """Fit SVI per (date, exdate, tau_bucket). Requires tau_mapped with tau_bucket."""
    df = tau_mapped.filter(pl.col("tau_bucket").is_not_null())
    if df.is_empty():
        return pl.DataFrame()

    results = []
    keys = df.select(["date", "exdate", "tau_bucket", "target_tau_days"]).unique()
    for i in range(len(keys)):
        row = keys.row(i, named=True)
        date_val, exdate_val, tau_bucket, target_tau = row["date"], row["exdate"], row["tau_bucket"], row["target_tau_days"]
        sub = df.filter(
            (pl.col("date") == date_val) & (pl.col("exdate") == exdate_val) &
            (pl.col("tau_bucket") == tau_bucket) & (pl.col("target_tau_days") == target_tau)
        )

        forward = sub["spx_price"].median() if "spx_price" in sub.columns else None
        if forward is None or forward <= 0:
            continue

        strike = sub["strike_price"].to_numpy() / 1000.0
        k = np.log(strike / forward)
        iv = sub["impl_volatility"].to_numpy().astype(float)
        vega = sub["vega"].to_numpy() if "vega" in sub.columns else None
        if vega is not None:
            vega = np.maximum(vega.astype(float), 1e-6)

        tau_days = (exdate_val - date_val).days if hasattr(exdate_val, "days") else target_tau
        tau = tau_days / 365.0

        valid = np.isfinite(iv) & (iv > 0) & np.isfinite(k)
        if valid.sum() < 5:
            continue

        k, iv = k[valid], iv[valid]
        vega = vega[valid] if vega is not None else None

        svi_cfg = config.get("svi", {})
        if svi_cfg.get("assert_units", False):
            assert_iv_bounds(iv, "IV_mkt")
            w_mkt = (iv ** 2) * tau
            assert_total_var_bounds(w_mkt, "w_mkt")

        fit = fit_svi_slice(k, iv, vega, tau, config)
        fit["date"] = date_val
        fit["exdate"] = exdate_val
        fit["tau_bucket"] = tau_bucket
        fit["target_tau_days"] = target_tau
        fit["forward"] = float(forward)
        fit["tau"] = tau
        fit["spx_price"] = float(forward)
        results.append(fit)

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results)
