"""Basic pipeline tests."""

import pytest
from pathlib import Path

from pipeline.config import load_config, compute_config_hash, validate_config


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
