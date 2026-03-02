"""Density diagnostics: nonnegativity, integral, mass outside grid, debug plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def check_density_diagnostics(
    k: np.ndarray,
    q: np.ndarray,
    date_str: str,
    tau_bucket: str,
    report_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Check density and optionally save debug plot on failure."""
    integral = float(np.trapezoid(q, k))
    nonneg = np.all(q >= -1e-10)
    clip_frac = np.mean(q < 0) if np.any(q < 0) else 0.0
    mass_err_max = config.get("fit", {}).get("density_mass_error_max", 0.01)
    invalid = abs(integral - 1.0) > mass_err_max or not nonneg

    result = {
        "q_integral": integral,
        "q_neg_clip_frac": clip_frac,
        "invalid_density": invalid,
    }

    if invalid and config.get("density", {}).get("debug_plot_on_fail", True):
        _save_debug_plot(k, q, date_str, tau_bucket, report_dir)

    return result


def _save_debug_plot(k: np.ndarray, q: np.ndarray, date_str: str, tau_bucket: str, report_dir: Path) -> None:
    """Save debug plot when density fails."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k, q, "b-")
        ax.axhline(0, color="k", ls="--")
        ax.set_xlabel("Log-moneyness k")
        ax.set_ylabel("Density q(k)")
        ax.set_title(f"Density debug: {date_str} {tau_bucket}")
        debug_dir = report_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(debug_dir / f"density_{date_str}_{tau_bucket}.png", dpi=100)
        plt.close(fig)
    except Exception:
        pass
