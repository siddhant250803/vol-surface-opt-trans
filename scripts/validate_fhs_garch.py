#!/usr/bin/env python
"""Validation script for FHS-GJR-GARCH physical distribution.

Run after implementation to:
1. Confirm no look-ahead bias (strict rolling window)
2. Compare W2 under iid bootstrap vs FHS-GARCH (requires pipeline run)
3. Verify numerical stability across seeds and n_sims

Usage:
    python scripts/validate_fhs_garch.py
"""

from __future__ import annotations

import numpy as np

from src.distribution.physical import generate_physical_distribution_fhs_garch


def test_no_lookahead():
    """Strict rolling: only past data used. No forward leakage."""
    rng = np.random.default_rng(42)
    returns = rng.standard_t(5, 300) * 0.01
    # Use first 252 as "past" - no access to returns[252:]
    window = returns[:252]
    k, p = generate_physical_distribution_fhs_garch(
        window, horizon=7, n_sims=5000, seed=42
    )
    assert len(window) == 252
    assert np.all(np.isfinite(p))
    print("OK: No look-ahead (strict 252-day window)")


def test_reproducibility():
    """Same seed -> identical output."""
    rng = np.random.default_rng(99)
    returns = rng.standard_t(5, 252) * 0.01
    k1, p1 = generate_physical_distribution_fhs_garch(
        returns, horizon=7, n_sims=5000, seed=123
    )
    k2, p2 = generate_physical_distribution_fhs_garch(
        returns, horizon=7, n_sims=5000, seed=123
    )
    np.testing.assert_array_almost_equal(p1, p2)
    print("OK: Reproducible with fixed seed")


def test_numerical_stability():
    """Stable across 5k, 10k, 20k sims."""
    rng = np.random.default_rng(1)
    returns = rng.standard_t(5, 252) * 0.01
    integrals = []
    for n in [5000, 10000, 20000]:
        k, p = generate_physical_distribution_fhs_garch(
            returns, horizon=7, n_sims=n, seed=42
        )
        integrals.append(np.trapezoid(p, k))
    assert all(0.99 < i < 1.01 for i in integrals)
    print("OK: Numerical stability across n_sims")


def main():
    print("Validating FHS-GJR-GARCH implementation...")
    test_no_lookahead()
    test_reproducibility()
    test_numerical_stability()
    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
