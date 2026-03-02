"""Experiment buckets: Q vs Q, Q vs P, model error."""

from src.experiments.q_vs_q import run_q_vs_q_experiment
from src.experiments.q_vs_p import run_q_vs_p_experiment

__all__ = ["run_q_vs_q_experiment", "run_q_vs_p_experiment"]
