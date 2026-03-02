"""Structured logging, git hash, seed logging."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path


def get_git_hash() -> str:
    """Return git commit hash or 'no-git' if not in repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "no-git"


def setup_logging(level: str = "INFO", log_config_hash: bool = True) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def log_run_context(seed: int, config_hash: str, dry_run: bool = False) -> None:
    """Log run context: seed, git hash, config hash."""
    logger = logging.getLogger("pipeline")
    logger.info("Run context: seed=%s, git=%s, config_hash=%s, dry_run=%s",
                seed, get_git_hash(), config_hash, dry_run)
