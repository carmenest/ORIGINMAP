"""
Robust Statistics Module for Mass Distribution Analysis.

Statistics implemented:
- CV: Coefficient of Variation (std/mean)
- varlog: Variance of log-transformed mass
- MAD_ratio: Median Absolute Deviation / Median

All statistics are designed to measure dispersion/heterogeneity
in mass distributions, with different sensitivity to outliers.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DispersionStats:
    """Container for all dispersion statistics."""
    cv: float
    varlog: float
    mad_ratio: float
    n: int

    def to_dict(self) -> Dict:
        return {
            "cv": self.cv,
            "varlog": self.varlog,
            "mad_ratio": self.mad_ratio,
            "n": self.n
        }


def coefficient_of_variation(masses: np.ndarray) -> float:
    """
    Coefficient of Variation: std / mean.

    Sensitive to outliers. Traditional measure.
    Returns 0 if mean is 0 or array is empty.
    """
    masses = np.asarray(masses)
    masses = masses[~np.isnan(masses)]

    if len(masses) < 2:
        return np.nan

    mean = np.mean(masses)
    if mean == 0:
        return np.nan

    return np.std(masses, ddof=1) / mean


def variance_of_log(masses: np.ndarray) -> float:
    """
    Variance of log-transformed mass: var(log(mass)).

    Robust to multiplicative outliers.
    Appropriate for log-normal distributions.
    Returns NaN if any mass <= 0.
    """
    masses = np.asarray(masses)
    masses = masses[~np.isnan(masses)]
    masses = masses[masses > 0]

    if len(masses) < 2:
        return np.nan

    return np.var(np.log(masses), ddof=1)


def mad_ratio(masses: np.ndarray) -> float:
    """
    MAD / Median: Median Absolute Deviation divided by Median.

    Highly robust to outliers.
    MAD = median(|x - median(x)|)

    Returns NaN if median is 0 or array is empty.
    """
    masses = np.asarray(masses)
    masses = masses[~np.isnan(masses)]

    if len(masses) < 2:
        return np.nan

    med = np.median(masses)
    if med == 0:
        return np.nan

    mad = np.median(np.abs(masses - med))
    return mad / med


def compute_all_stats(masses: np.ndarray) -> DispersionStats:
    """
    Compute all dispersion statistics for a mass array.

    Returns DispersionStats dataclass with cv, varlog, mad_ratio, n.
    """
    masses = np.asarray(masses)
    masses = masses[~np.isnan(masses)]

    return DispersionStats(
        cv=coefficient_of_variation(masses),
        varlog=variance_of_log(masses),
        mad_ratio=mad_ratio(masses),
        n=len(masses)
    )


def bootstrap_stats(
    masses: np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 42,
    confidence: float = 0.95
) -> Dict[str, Tuple[float, float, float]]:
    """
    Bootstrap confidence intervals for all statistics.

    Returns dict with keys 'cv', 'varlog', 'mad_ratio'.
    Each value is (observed, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    masses = np.asarray(masses)
    masses = masses[~np.isnan(masses)]

    n = len(masses)
    if n < 10:
        # Not enough data for reliable bootstrap
        obs = compute_all_stats(masses)
        return {
            "cv": (obs.cv, np.nan, np.nan),
            "varlog": (obs.varlog, np.nan, np.nan),
            "mad_ratio": (obs.mad_ratio, np.nan, np.nan)
        }

    # Bootstrap samples
    cv_boot = []
    varlog_boot = []
    mad_boot = []

    for _ in range(n_bootstrap):
        sample = rng.choice(masses, size=n, replace=True)
        cv_boot.append(coefficient_of_variation(sample))
        varlog_boot.append(variance_of_log(sample))
        mad_boot.append(mad_ratio(sample))

    alpha = (1 - confidence) / 2

    def ci(values, observed):
        values = np.array(values)
        values = values[~np.isnan(values)]
        if len(values) < 10:
            return (observed, np.nan, np.nan)
        return (
            observed,
            np.percentile(values, alpha * 100),
            np.percentile(values, (1 - alpha) * 100)
        )

    obs = compute_all_stats(masses)

    return {
        "cv": ci(cv_boot, obs.cv),
        "varlog": ci(varlog_boot, obs.varlog),
        "mad_ratio": ci(mad_boot, obs.mad_ratio)
    }


# Convenience functions for use in null models
def cv(masses: np.ndarray) -> float:
    """Alias for coefficient_of_variation."""
    return coefficient_of_variation(masses)


def varlog(masses: np.ndarray) -> float:
    """Alias for variance_of_log."""
    return variance_of_log(masses)


def mad(masses: np.ndarray) -> float:
    """Alias for mad_ratio."""
    return mad_ratio(masses)


# Stat function registry for dynamic selection
STAT_FUNCTIONS = {
    "cv": coefficient_of_variation,
    "varlog": variance_of_log,
    "mad_ratio": mad_ratio,
    "mad": mad_ratio,  # alias
}


def get_stat_function(name: str):
    """Get statistic function by name."""
    if name not in STAT_FUNCTIONS:
        raise ValueError(f"Unknown statistic: {name}. Available: {list(STAT_FUNCTIONS.keys())}")
    return STAT_FUNCTIONS[name]
