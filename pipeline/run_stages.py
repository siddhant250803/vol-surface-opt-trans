"""Orchestrate pipeline stages: load -> clean -> tau_map -> svi_fit -> q_recover -> p_estimate -> distances -> experiments -> report."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from src.data.load_options import load_options
from src.data.load_rates import load_rates
from src.data.clean import clean_options
from src.surface.tau_mapping import map_tau_buckets
from src.surface.svi_fit import fit_svi_surface
from src.surface.call_prices import compute_call_prices
from src.distribution.risk_neutral import recover_risk_neutral_density
from src.distribution.physical import estimate_physical_density
from src.distribution.constant_maturity import build_constant_maturity_q
from src.distances.wasserstein import compute_all_distances
from src.experiments.q_vs_q import run_q_vs_q_experiment
from src.experiments.q_vs_p import run_q_vs_p_experiment
from src.experiments.report import generate_all_charts

logger = logging.getLogger("pipeline")


def run_all_stages(config: dict[str, Any], project_root: Path, config_hash: str, skip_cache: bool) -> None:
    """Run all pipeline stages in order."""
    processed_dir = project_root / "data" / "processed" / config_hash
    cache_dir = project_root / "outputs" / "cache" / config_hash
    report_dir = project_root / "outputs" / "report"
    features_dir = project_root / "outputs" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config.get("data", {})
    opts_path = project_root / data_cfg.get("options_path", "")
    rates_path = project_root / data_cfg.get("rates_path", "")

    # Stage 1: Load
    logger.info("Stage 1: Load options and rates")
    options = load_options(str(opts_path), config)
    rates = load_rates(str(rates_path), config)

    config_full_range = {**config, "data": {**config.get("data", {}), "anchor_dates": [], "include_neighbor_dates": False}}
    options_full = load_options(str(opts_path), config_full_range)

    # Stage 2: Clean
    logger.info("Stage 2: Clean options")
    options_clean = clean_options(options, config)
    options_clean.write_parquet(processed_dir / "options_clean.parquet")

    # Stage 3: Tau mapping
    logger.info("Stage 3: Tau mapping")
    tau_mapped = map_tau_buckets(options_clean, config)
    tau_mapped.write_parquet(processed_dir / "tau_mapped.parquet")

    # Stage 4: SVI fit
    logger.info("Stage 4: SVI fit")
    svi_results = fit_svi_surface(tau_mapped, rates, config)
    svi_results.write_parquet(cache_dir / "svi_results.parquet")

    # Stage 5: Call prices
    logger.info("Stage 5: Call prices")
    call_prices = compute_call_prices(svi_results, config)
    call_prices.write_parquet(cache_dir / "call_prices.parquet")

    # Stage 6: Q recovery
    logger.info("Stage 6: Q recovery")
    q_densities = recover_risk_neutral_density(call_prices, config)
    q_densities.write_parquet(cache_dir / "q_densities.parquet")

    # Stage 7: P estimation (use full options for return history, target dates from clean)
    logger.info("Stage 7: P estimation")
    target_dates = options_clean.select("date").unique().sort("date")["date"].to_list()
    p_densities = estimate_physical_density(options_full, rates, config, target_dates=target_dates)
    p_densities.write_parquet(cache_dir / "p_densities.parquet")

    # Stage 8: Distances
    logger.info("Stage 8: Distances")
    q_constant_maturity = None
    if config.get("distances", {}).get("use_constant_maturity_q_prev", False):
        q_constant_maturity = build_constant_maturity_q(svi_results, config)
    distances = compute_all_distances(q_densities, p_densities, config, q_constant_maturity)
    distances.write_parquet(cache_dir / "distances.parquet")

    # Stage 9: Experiments
    logger.info("Stage 9: Experiments")
    features = run_q_vs_q_experiment(distances, svi_results, config)
    features = run_q_vs_p_experiment(features, distances, options_clean, config)
    features.write_parquet(features_dir / f"features_{config_hash}.parquet")

    # Stage 10: Report
    logger.info("Stage 10: Report")
    generate_all_charts(features, config, report_dir)

    # Stage 11: Golden (if enabled)
    if config.get("golden", {}).get("enabled", False):
        logger.info("Stage 11: Golden test")
        from src.experiments.golden import run_golden_assertions, generate_golden_artifacts
        features = run_golden_assertions(
            options_clean, tau_mapped, svi_results, call_prices,
            q_densities, p_densities, distances, features, rates, config,
        )
        features.write_parquet(features_dir / f"features_{config_hash}.parquet")
        generate_golden_artifacts(
            tau_mapped, svi_results, call_prices, q_densities, p_densities,
            distances, features, options_clean, config, report_dir, config_hash,
        )

    logger.info("Pipeline complete.")
