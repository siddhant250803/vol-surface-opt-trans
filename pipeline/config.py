"""Load, merge, validate config; compute config hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and merge YAML configs. Base config can extend other configs."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    base_dir = config_path.parent
    with open(config_path) as f:
        config = yaml.safe_load(f)

    extends = config.pop("extends", [])
    for ext in extends:
        ext_path = (base_dir / f"{ext}.yaml").resolve()
        if ext_path.exists():
            with open(ext_path) as f:
                ext_config = yaml.safe_load(f)
            config = _deep_merge(ext_config or {}, config)
        else:
            raise FileNotFoundError(f"Extended config not found: {ext_path}")

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def compute_config_hash(config: dict[str, Any]) -> str:
    """Compute SHA256 hash of config for cache keying."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def validate_config(config: dict[str, Any], project_root: Path) -> list[str]:
    """Validate config and data paths. Returns list of error messages."""
    errors = []

    if "data" not in config:
        errors.append("Missing 'data' section")
    else:
        data = config["data"]
        opts_path = project_root / data.get("options_path", "")
        if not opts_path.exists():
            errors.append(f"Options path not found: {opts_path}")
        rates_path = project_root / data.get("rates_path", "")
        if not rates_path.exists():
            errors.append(f"Rates path not found: {rates_path}")

    if not config.get("tau_buckets"):
        errors.append("Missing or empty 'tau_buckets' section")

    if "fit" not in config:
        errors.append("Missing 'fit' section with iv_rmse_max and density_mass_error_max")

    return errors
