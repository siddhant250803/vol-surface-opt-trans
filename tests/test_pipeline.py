"""Basic pipeline tests."""

import numpy as np
import pytest
from pathlib import Path

from pipeline.config import load_config, compute_config_hash, validate_config


def test_fhs_garch_reproducibility():
    """FHS-GARCH is reproducible with fixed seed."""
    from src.distribution.physical import generate_physical_distribution_fhs_garch

    rng = np.random.default_rng(42)
    returns = rng.standard_t(5, 252) * 0.01
    k1, p1 = generate_physical_distribution_fhs_garch(
        returns, horizon=7, n_sims=2000, seed=42
    )
    k2, p2 = generate_physical_distribution_fhs_garch(
        returns, horizon=7, n_sims=2000, seed=42
    )
    np.testing.assert_array_almost_equal(p1, p2)
    assert np.trapezoid(p1, k1) > 0.99


def test_fhs_garch_numerical_stability():
    """FHS-GARCH is stable across simulation counts."""
    from src.distribution.physical import generate_physical_distribution_fhs_garch

    rng = np.random.default_rng(123)
    returns = rng.standard_t(5, 252) * 0.01
    k5k, p5k = generate_physical_distribution_fhs_garch(
        returns, horizon=7, n_sims=5000, seed=1
    )
    k10k, p10k = generate_physical_distribution_fhs_garch(
        returns, horizon=7, n_sims=10000, seed=1
    )
    # Same seed, more sims: similar shape (allow some variance)
    assert np.trapezoid(p5k, k5k) > 0.99
    assert np.trapezoid(p10k, k10k) > 0.99
    assert np.abs(np.mean(p5k) - np.mean(p10k)) < 0.5  # rough sanity


def test_load_config():
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "configs/base.yaml")
    assert "seed" in config
    assert config["seed"] == 42
    assert "data" in config
    assert "tau_buckets" in config


def test_config_hash():
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "configs/base.yaml")
    h = compute_config_hash(config)
    assert len(h) == 16
    assert h.isalnum()


def test_validate_config():
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "configs/base.yaml")
    errors = validate_config(config, project_root)
    assert len(errors) == 0
