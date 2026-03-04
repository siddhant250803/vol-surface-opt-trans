"""Physical distribution P: FHS-GJR-GARCH from historical returns.

Replaces the previous iid bootstrap with Filtered Historical Simulation (FHS)
using a GJR-GARCH(1,1) leverage model. Residual bootstrap preserves volatility
clustering and asymmetry; the leverage term captures the empirically observed
effect that negative returns amplify future volatility more than positive ones.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
from arch import arch_model
from scipy import stats

logger = logging.getLogger(__name__)


def _get_log_returns(options: pl.DataFrame) -> pl.DataFrame:
    """Extract daily log returns from SPX prices."""
    if "spx_price" not in options.columns:
        return pl.DataFrame()

    prices = options.select(["date", "spx_price"]).unique().sort("date")
    prices = prices.with_columns(pl.col("spx_price").log().alias("log_price"))
    prices = prices.with_columns(pl.col("log_price").diff().alias("log_return"))
    return prices.filter(pl.col("log_return").is_not_null())


def generate_physical_distribution_fhs_garch(
    returns_window: np.ndarray,
    horizon: int = 7,
    n_sims: int = 10000,
    k_grid: np.ndarray | None = None,
    seed: int | None = None,
    garch_model: str = "GJR",
    dist: str = "t",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate physical distribution P_t via FHS with GJR-GARCH(1,1).

    Filtered Historical Simulation (FHS) uses a fitted GARCH model to obtain
    standardized residuals, then bootstraps those (not raw returns) and
    forward-simulates with the GARCH volatility recursion. This preserves
    volatility clustering and fat tails while avoiding the look-ahead bias
    of iid return bootstrap.

    Why residual bootstrap > iid return bootstrap:
    - Raw returns are heteroskedastic; iid bootstrap assumes constant variance.
    - FHS resamples standardized residuals ε_t = r_t/σ_t, which are approximately
      iid under the model, then scales by simulated σ_{t+h}. This correctly
      propagates volatility dynamics to the forecast horizon.

    Why include the leverage term (GJR-GARCH):
    - Equity returns exhibit asymmetric volatility: negative shocks increase
      future volatility more than positive shocks of equal magnitude (leverage effect).
    - GJR-GARCH adds γ·I(r<0)·r² to the variance equation, capturing this.
    - EGARCH is an alternative; GJR is simpler and widely used.

    Parameters
    ----------
    returns_window : np.ndarray
        Past 252 (or window_days) daily log returns. Strict rolling, no forward data.
    horizon : int
        Forward simulation horizon in days (e.g. 7 for 7-day cumulative return).
    n_sims : int
        Number of bootstrap paths (≥5000 recommended for stability).
    k_grid : np.ndarray | None
        Log-moneyness grid for density evaluation. If None, uses default [-0.5, 0.5]
        with 201 points.
    seed : int | None
        Random seed for reproducibility. If None, uses current RNG state.
    garch_model : str
        "GJR" for GJR-GARCH(1,1) (default), "EGARCH" for EGARCH(1,1).
    dist : str
        Innovation distribution: "t" (Student-t, default) or "normal".

    Returns
    -------
    k_grid : np.ndarray
        Log-moneyness grid where density is evaluated.
    p_density : np.ndarray
        Density on k_grid, normalized to integrate to 1.
    """
    if k_grid is None:
        k_grid = np.linspace(-0.5, 0.5, 201)

    n = len(returns_window)
    logger.debug("generate_physical_distribution_fhs_garch: n=%d, horizon=%d, n_sims=%d", n, horizon, n_sims)
    if n < 100:
        # Fallback: return flat density if insufficient data
        logger.warning("Insufficient returns (n=%d < 100), returning flat density", n)
        p_density = np.ones_like(k_grid, dtype=float)
        p_density = p_density / (np.trapezoid(p_density, k_grid) or 1e-10)
        return k_grid, p_density

    rng = np.random.default_rng(seed)

    # arch expects percentage returns for numerical stability
    y = returns_window * 100.0
    logger.debug("Returns: mean=%.4f%%, std=%.4f%%, min=%.4f, max=%.4f", y.mean(), y.std(), y.min(), y.max())

    if garch_model == "EGARCH":
        am = arch_model(y, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist=dist)
    else:
        # GJR-GARCH: p=1, o=1, q=1 with GARCH vol gives asymmetric (GJR) model
        am = arch_model(y, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist=dist)

    try:
        res = am.fit(disp="off", show_warning=False)
        logger.debug("GARCH fit OK: omega=%.6f alpha=%.4f gamma=%.4f beta=%.4f",
                     float(res.params.get("omega", 0)), float(res.params.get("alpha[1]", 0)),
                     float(res.params.get("gamma[1]", 0)), float(res.params.get("beta[1]", 0)))
    except Exception as e:
        logger.warning("GARCH fit failed: %s, returning flat density", e)
        p_density = np.ones_like(k_grid, dtype=float)
        p_density = p_density / (np.trapezoid(p_density, k_grid) or 1e-10)
        return k_grid, p_density

    # Standardized residuals: ε_t = r_t / σ_t (approximately iid under the model)
    cond_vol = res.conditional_volatility
    std_resid = res.std_resid
    std_resid = std_resid[np.isfinite(std_resid)]
    logger.debug("std_resid: len=%d, mean=%.4f, std=%.4f", len(std_resid), float(np.mean(std_resid)), float(np.std(std_resid)))
    if len(std_resid) < 50:
        p_density = np.ones_like(k_grid, dtype=float)
        p_density = p_density / (np.trapezoid(p_density, k_grid) or 1e-10)
        return k_grid, p_density

    params = res.params
    sigma2 = float(cond_vol[-1] ** 2)  # %²
    n_resid = len(std_resid)

    if garch_model == "EGARCH":
        # EGARCH(1,1): log(σ²_{t+1}) = ω + α*(|z|-E|z|) + γ*z + β*log(σ²_t)
        omega = float(params.get("omega", 1e-6))
        alpha = float(params.get("alpha[1]", 0.1))
        gamma = float(params.get("gamma[1]", 0.0))
        beta = float(params.get("beta[1]", 0.85))
        # E|z| for standard normal; Student-t uses different value
        e_abs_z = np.sqrt(2 / np.pi) if dist == "normal" else 1.0

        def _update_sigma2(s2: float, z: float) -> float:
            log_s2 = np.log(max(s2, 1e-12))
            log_s2_new = omega + alpha * (np.abs(z) - e_abs_z) + gamma * z + beta * log_s2
            return max(np.exp(log_s2_new), 1e-12)
    else:
        # GJR-GARCH(1,1): σ²_{t+1} = ω + α·r² + γ·I(r<0)·r² + β·σ²
        omega = float(params.get("omega", params.get("Constant", 1e-6)))
        alpha = float(params.get("alpha[1]", 0.1))
        gamma = float(params.get("gamma[1]", 0.0))
        beta = float(params.get("beta[1]", 0.85))

        def _update_sigma2(s2: float, r_pct: float) -> float:
            r_sq = r_pct**2
            ind_neg = 1.0 if r_pct < 0 else 0.0
            return max(omega + alpha * r_sq + gamma * ind_neg * r_sq + beta * s2, 1e-12)

    cum_returns = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        sigma2_h = sigma2
        path_sum = 0.0

        for _ in range(horizon):
            idx = rng.integers(0, n_resid)
            eps_star = std_resid[idx]
            r_pct = np.sqrt(sigma2_h) * eps_star
            path_sum += r_pct
            if garch_model == "EGARCH":
                sigma2_h = _update_sigma2(sigma2_h, eps_star)
            else:
                sigma2_h = _update_sigma2(sigma2_h, r_pct)

        cum_returns[i] = path_sum / 100.0  # convert % to decimal log-return

    # Convert simulated 7-day cumulative log returns to density on k_grid
    # Cumulative log return = log(S_T/S_0) = log-moneyness
    k_min, k_max = float(k_grid.min()), float(k_grid.max())
    cum_returns = np.clip(cum_returns, k_min - 0.01, k_max + 0.01)

    # KDE for smooth density (avoids histogram binning artifacts)
    logger.debug("cum_returns: mean=%.6f, std=%.6f, [1%%=%.4f, 99%%=%.4f]",
                 cum_returns.mean(), cum_returns.std(), np.percentile(cum_returns, 1), np.percentile(cum_returns, 99))
    try:
        kde = stats.gaussian_kde(cum_returns, bw_method="scott")
        p_density = kde(k_grid)
        logger.debug("KDE OK, integral=%.6f", float(np.trapezoid(p_density, k_grid)))
    except Exception as e:
        logger.warning("KDE failed: %s, using histogram fallback", e)
        # Fallback to histogram if KDE fails
        hist, bin_edges = np.histogram(
            cum_returns, bins=len(k_grid), range=(k_min, k_max), density=True
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        p_density = np.interp(k_grid, bin_centers, hist)

    p_density = np.maximum(p_density, 1e-12)
    integral = np.trapezoid(p_density, k_grid)
    if integral > 1e-10:
        p_density = p_density / integral

    return k_grid, p_density


def estimate_physical_density(
    options: pl.DataFrame,
    rates: pl.DataFrame,
    config: dict[str, Any],
    target_dates: list | None = None,
) -> pl.DataFrame:
    """Estimate P for each (date, tau) via FHS-GJR-GARCH.

    Replaces the previous iid bootstrap with Filtered Historical Simulation
    using GJR-GARCH(1,1). Uses strict 252-day rolling window, no look-ahead.
    """
    phys_cfg = config.get("physical", {})
    window_days = phys_cfg.get("window_days", 252)
    n_sims = phys_cfg.get("n_sims", phys_cfg.get("bootstrap_n", 10000))
    seed = config.get("seed", None)
    garch_model = phys_cfg.get("garch_model", "GJR")
    dist = phys_cfg.get("garch_dist", "t")

    returns_df = _get_log_returns(options)
    if returns_df.is_empty():
        logger.warning("estimate_physical_density: no log returns (spx_price missing or empty)")
        return pl.DataFrame()

    returns = returns_df["log_return"].to_numpy().astype(float)
    dates = returns_df["date"].to_numpy()
    logger.info("P estimation: %d returns, %d unique dates, window=%d, n_sims=%d, model=%s",
                len(returns), len(np.unique(dates)), window_days, n_sims, garch_model)

    k_cfg = config.get("density", {}).get("k_grid", {})
    n_points = k_cfg.get("n_points", 201)
    k_min = k_cfg.get("k_min", -0.5)
    k_max = k_cfg.get("k_max", 0.5)
    k_grid = np.linspace(k_min, k_max, n_points)

    unique_dates = (
        target_dates
        if target_dates is not None
        else options.select("date").unique().sort("date")["date"].to_list()
    )
    buckets = config.get("tau_buckets", [])
    if not buckets:
        return pl.DataFrame()
    tau_buckets = [
        {
            "tau_bucket": b.get("label", str(b["target_days"])),
            "target_tau_days": b["target_days"],
        }
        for b in buckets
    ]

    rows = []
    n_dates_ok = 0
    for date_val in unique_dates:
        idx = np.where(dates <= date_val)[0]
        if len(idx) < window_days:
            continue
        n_dates_ok += 1
        window_returns = returns[idx[-window_days]:]

        for tb_row in tau_buckets:
            tau_bucket = tb_row.get("tau_bucket", str(tb_row["target_tau_days"]))
            target_tau = tb_row["target_tau_days"]
            horizon = max(1, int(target_tau))

            k_out, p_density = generate_physical_distribution_fhs_garch(
                returns_window=window_returns,
                horizon=horizon,
                n_sims=n_sims,
                k_grid=k_grid,
                seed=seed,
                garch_model=garch_model,
                dist=dist,
            )

            rows.append({
                "date": date_val,
                "tau_bucket": tau_bucket,
                "target_tau_days": target_tau,
                "k_grid": k_out.tolist(),
                "p_density": p_density.tolist(),
                "p_integral": float(np.trapezoid(p_density, k_out)),
            })
        if n_dates_ok <= 3 or n_dates_ok % 50 == 0:
            logger.info("P estimation progress: %d dates done, %d rows so far", n_dates_ok, len(rows))

    logger.info("P estimation complete: %d rows (dates x tau_buckets)", len(rows))
    return pl.DataFrame(rows)
