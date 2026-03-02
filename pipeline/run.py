"""Entrypoint: python -m pipeline.run --config configs/base.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.config import load_config, compute_config_hash, validate_config
from pipeline.logging_utils import setup_logging, log_run_context


def main() -> None:
    parser = argparse.ArgumentParser(description="SPX Vol Surface Pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Config YAML path")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and paths only")
    parser.add_argument("--skip-cache", action="store_true", help="Recompute all, overwrite cache")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / args.config

    config = load_config(config_path)
    config_hash = compute_config_hash(config)

    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_config_hash=log_cfg.get("log_config_hash", True),
    )
    logger = __import__("logging").getLogger("pipeline")

    seed = config.get("seed", 42)
    np.random.seed(seed)
    log_run_context(seed, config_hash, dry_run=args.dry_run)

    errors = validate_config(config, project_root)
    if errors:
        for e in errors:
            logger.error("Config validation failed: %s", e)
        raise SystemExit(1)

    if args.dry_run:
        logger.info("Dry run complete. Config and paths OK.")
        return

    processed_dir = project_root / "data" / "processed" / config_hash
    cache_dir = project_root / "outputs" / "cache" / config_hash
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processed dir: %s", processed_dir)
    logger.info("Cache dir: %s", cache_dir)

    from pipeline.run_stages import run_all_stages
    run_all_stages(config, project_root, config_hash, args.skip_cache)


if __name__ == "__main__":
    main()
