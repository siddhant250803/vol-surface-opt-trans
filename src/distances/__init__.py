"""Distribution distance metrics: Wasserstein, quantile gaps, moments."""

from src.distances.wasserstein import wasserstein_1, wasserstein_2
from src.distances.quantile_gaps import quantile_gap
from src.distances.moments import tail_mass_gap, moment_diff

__all__ = [
    "wasserstein_1",
    "wasserstein_2",
    "quantile_gap",
    "tail_mass_gap",
    "moment_diff",
]
