#!/usr/bin/env python
"""Test P estimation with full logging. Run: PYTHONPATH=. python scripts/test_p_estimation.py"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Configure logging to stdout, DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Ensure our module logs
logging.getLogger("src.distribution.physical").setLevel(logging.DEBUG)

import polars as pl

from pipeline.config import load_config
from src.distribution.physical import estimate_physical_density, generate_physical_distribution_fhs_garch
from src.data.load_options import load_options
from src.data.load_rates import load_rates


def main():
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "configs/base.yaml")

    # Override for fast test: fewer sims, fewer dates
    config["physical"] = config.get("physical", {}) | {
        "n_sims": 500,
        "window_days": 252,
    }

    print("=" * 60)
    print("1. Unit test: generate_physical_distribution_fhs_garch")
    print("=" * 60)
    import numpy as np
    rng = np.random.default_rng(42)
    returns = rng.standard_t(5, 252) * 0.01
    k, p = generate_physical_distribution_fhs_garch(
        returns, horizon=7, n_sims=500, seed=42
    )
    integral = float(np.trapezoid(p, k))
    print(f"  k shape: {k.shape}, p shape: {p.shape}")
    print(f"  p integral: {integral:.6f}")
    print(f"  p min/max: {p.min():.6f} / {p.max():.6f}")
    assert 0.99 < integral < 1.01, f"Integral should be ~1, got {integral}"
    print("  OK: unit test passed\n")

    print("=" * 60)
    print("2. Integration test: estimate_physical_density on real data")
    print("=" * 60)

    data_cfg = config.get("data", {})
    opts_path = project_root / data_cfg.get("options_path", "")
    rates_path = project_root / data_cfg.get("rates_path", "")

    if not opts_path.exists():
        print(f"  SKIP: options path not found: {opts_path}")
        return

    options_full = load_options(str(opts_path), config)
    rates = load_rates(str(rates_path), config)

    # Use dates that have 252-day history (skip first ~252 trading days)
    all_dates = options_full.select("date").unique().sort("date")["date"].to_list()
    # Need window_days of returns before first valid date
    target_dates = all_dates[260:265]  # ~1 year in, 5 dates for fast test
    print(f"  target_dates (5 from mid-series): {target_dates}")

    p_densities = estimate_physical_density(
        options_full, rates, config, target_dates=target_dates
    )

    if p_densities.is_empty():
        print("  FAIL: p_densities is empty")
        return

    print(f"  p_densities shape: {p_densities.shape}")
    print(f"  columns: {p_densities.columns}")
    print(f"  date x tau_bucket sample:")
    for row in p_densities.head(3).iter_rows(named=True):
        print(f"    {row['date']} {row['tau_bucket']}: p_integral={row['p_integral']:.6f}")

    # Sanity: all integrals should be ~1
    integrals = p_densities["p_integral"].to_numpy()
    bad = (integrals < 0.99) | (integrals > 1.01)
    if bad.any():
        print(f"  WARN: {bad.sum()} rows have p_integral outside [0.99, 1.01]")
    else:
        print("  OK: all p_integral in [0.99, 1.01]")

    print("\nP estimation test complete.")


if __name__ == "__main__":
    main()
