"""
O-Δ10: Falls-Only Analysis — The Unbiased Sample

Key insight: Falls (meteorites we saw fall) have NO collection bias.
- No size bias (we didn't search for them)
- No color/visibility bias
- No location bias (they fell where they fell)

This is our best window into the TRUE flux of meteorites to Earth.

Hypotheses:
- H-FALL-1: Falls show genuine mass structure (survives Null-5)
- H-FALL-2: Class proportions differ between Falls and Finds
- H-FALL-3: Falls show seasonality (orbital signature)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy import stats
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from originmap.config import DATA_PROCESSED, REPORTS


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


@dataclass
class FallsProfile:
    """Summary statistics for Falls subset."""
    n_total: int
    n_with_mass: int
    n_with_year: int
    n_classes: int
    mass_median: float
    mass_mean: float
    mass_log_mean: float
    mass_log_std: float
    year_range: Tuple[int, int]
    top_classes: List[Tuple[str, int]]


def compute_falls_profile(df: pd.DataFrame) -> FallsProfile:
    """Compute profile of Falls data."""
    n_total = len(df)
    n_with_mass = df["mass"].notna().sum()
    n_with_year = df["year"].notna().sum()

    class_counts = df["recclass"].value_counts()
    n_classes = len(class_counts)

    masses = df["mass"].dropna()
    masses = masses[masses > 0]

    if len(masses) > 0:
        mass_median = float(np.median(masses))
        mass_mean = float(np.mean(masses))
        log_masses = np.log(masses)
        mass_log_mean = float(np.mean(log_masses))
        mass_log_std = float(np.std(log_masses))
    else:
        mass_median = mass_mean = mass_log_mean = mass_log_std = np.nan

    years = df["year"].dropna()
    if len(years) > 0:
        year_range = (int(years.min()), int(years.max()))
    else:
        year_range = (0, 0)

    top_classes = [(c, int(n)) for c, n in class_counts.head(10).items()]

    return FallsProfile(
        n_total=n_total,
        n_with_mass=n_with_mass,
        n_with_year=n_with_year,
        n_classes=n_classes,
        mass_median=mass_median,
        mass_mean=mass_mean,
        mass_log_mean=mass_log_mean,
        mass_log_std=mass_log_std,
        year_range=year_range,
        top_classes=top_classes,
    )


def test_h_fall_1_mass_structure(
    falls_df: pd.DataFrame,
    n_permutations: int = 1000,
    subsample_size: int = 30,
    seed: int = 42
) -> Dict[str, Any]:
    """
    H-FALL-1: Test if Falls show genuine mass structure.

    Uses balanced subsampling (Null-5) to test if mass heterogeneity
    in Falls classes is real or artifact.
    """
    np.random.seed(seed)

    # Filter to classes with enough samples
    df = falls_df.dropna(subset=["mass", "recclass"]).copy()
    df = df[df["mass"] > 0]

    class_counts = df["recclass"].value_counts()
    valid_classes = class_counts[class_counts >= subsample_size].index.tolist()

    if len(valid_classes) < 3:
        return {
            "test": "h_fall_1",
            "status": "insufficient_data",
            "message": f"Only {len(valid_classes)} classes have >= {subsample_size} samples",
            "valid_classes": valid_classes,
        }

    df = df[df["recclass"].isin(valid_classes)]

    # Compute observed CV for each class (balanced subsample)
    def compute_balanced_cv(data: pd.DataFrame, classes: List[str], n: int) -> Dict[str, float]:
        cvs = {}
        for c in classes:
            class_data = data[data["recclass"] == c]["mass"].values
            if len(class_data) >= n:
                sample = np.random.choice(class_data, size=n, replace=False)
                log_sample = np.log(sample)
                cvs[c] = float(np.std(log_sample) / np.mean(log_sample)) if np.mean(log_sample) != 0 else np.nan
        return cvs

    # Observed CVs
    observed_cvs = compute_balanced_cv(df, valid_classes, subsample_size)

    # Null distribution: permute class labels, then compute CVs
    null_cv_spread = []  # Spread of CVs across classes under null

    for _ in range(n_permutations):
        # Permute class labels
        perm_df = df.copy()
        perm_df["recclass"] = np.random.permutation(perm_df["recclass"].values)

        null_cvs = compute_balanced_cv(perm_df, valid_classes, subsample_size)
        if null_cvs:
            null_cv_spread.append(np.std(list(null_cvs.values())))

    # Observed spread
    observed_spread = np.std(list(observed_cvs.values()))

    # P-value: how often does null show MORE spread than observed?
    # If classes have genuine structure, observed spread should be HIGH
    p_value = np.mean([ns >= observed_spread for ns in null_cv_spread])

    # Also test individual classes
    class_results = []
    for c in valid_classes:
        class_data = df[df["recclass"] == c]["mass"].values
        log_data = np.log(class_data)

        # Bootstrap CI for CV
        bootstrap_cvs = []
        for _ in range(500):
            sample = np.random.choice(log_data, size=min(len(log_data), subsample_size), replace=True)
            if np.mean(sample) != 0:
                bootstrap_cvs.append(np.std(sample) / abs(np.mean(sample)))

        if bootstrap_cvs:
            class_results.append({
                "class": c,
                "n": len(class_data),
                "cv_observed": observed_cvs.get(c, np.nan),
                "cv_bootstrap_mean": float(np.mean(bootstrap_cvs)),
                "cv_bootstrap_std": float(np.std(bootstrap_cvs)),
                "cv_ci_low": float(np.percentile(bootstrap_cvs, 2.5)),
                "cv_ci_high": float(np.percentile(bootstrap_cvs, 97.5)),
            })

    # Verdict
    if p_value < 0.05:
        verdict = "STRUCTURE_DETECTED"
        interpretation = "Falls show genuine mass heterogeneity differences between classes"
    else:
        verdict = "NO_STRUCTURE"
        interpretation = "Mass heterogeneity in Falls is consistent with random variation"

    return {
        "test": "h_fall_1",
        "status": "completed",
        "n_classes": len(valid_classes),
        "subsample_size": subsample_size,
        "n_permutations": n_permutations,
        "observed_cv_spread": float(observed_spread),
        "null_cv_spread_mean": float(np.mean(null_cv_spread)),
        "null_cv_spread_std": float(np.std(null_cv_spread)),
        "p_value": float(p_value),
        "verdict": verdict,
        "interpretation": interpretation,
        "class_results": class_results,
        "observed_cvs": {k: float(v) for k, v in observed_cvs.items()},
    }


def test_h_fall_2_class_proportions(
    falls_df: pd.DataFrame,
    finds_df: pd.DataFrame,
    min_count: int = 10
) -> Dict[str, Any]:
    """
    H-FALL-2: Test if class proportions differ between Falls and Finds.

    If certain classes are over/under-represented in Falls, this reveals
    something about their physical properties (survival during fall,
    visibility, fragmentation, etc.)
    """
    # Get class proportions
    falls_counts = falls_df["recclass"].value_counts()
    finds_counts = finds_df["recclass"].value_counts()

    falls_total = len(falls_df)
    finds_total = len(finds_df)

    # Find classes present in both with sufficient counts
    common_classes = set(falls_counts.index) & set(finds_counts.index)
    valid_classes = [c for c in common_classes
                    if falls_counts.get(c, 0) >= min_count and finds_counts.get(c, 0) >= min_count]

    if len(valid_classes) < 5:
        return {
            "test": "h_fall_2",
            "status": "insufficient_data",
            "message": f"Only {len(valid_classes)} classes meet minimum count threshold",
        }

    # Compute enrichment ratios
    enrichment = []
    for c in valid_classes:
        falls_n = falls_counts.get(c, 0)
        finds_n = finds_counts.get(c, 0)

        falls_frac = falls_n / falls_total
        finds_frac = finds_n / finds_total

        # Enrichment in Falls relative to Finds
        ratio = falls_frac / finds_frac if finds_frac > 0 else np.inf

        # Fisher's exact test for this class
        # Contingency table: [[falls_c, falls_other], [finds_c, finds_other]]
        table = [
            [falls_n, falls_total - falls_n],
            [finds_n, finds_total - finds_n]
        ]
        odds_ratio, fisher_p = stats.fisher_exact(table)

        enrichment.append({
            "class": c,
            "falls_n": int(falls_n),
            "finds_n": int(finds_n),
            "falls_frac": float(falls_frac),
            "finds_frac": float(finds_frac),
            "enrichment_ratio": float(ratio),
            "odds_ratio": float(odds_ratio),
            "fisher_p": float(fisher_p),
        })

    enrich_df = pd.DataFrame(enrichment)

    # Apply Benjamini-Hochberg correction
    enrich_df = enrich_df.sort_values("fisher_p")
    n = len(enrich_df)
    enrich_df["rank"] = range(1, n + 1)
    enrich_df["bh_threshold"] = enrich_df["rank"] * 0.05 / n
    enrich_df["significant"] = enrich_df["fisher_p"] < enrich_df["bh_threshold"]

    # Find over and under-represented classes
    sig_classes = enrich_df[enrich_df["significant"]]
    over_represented = sig_classes[sig_classes["enrichment_ratio"] > 1].sort_values("enrichment_ratio", ascending=False)
    under_represented = sig_classes[sig_classes["enrichment_ratio"] < 1].sort_values("enrichment_ratio")

    # Chi-square test on full distribution
    # Use only valid classes
    falls_obs = np.array([falls_counts.get(c, 0) for c in valid_classes])
    finds_obs = np.array([finds_counts.get(c, 0) for c in valid_classes])

    # Expected under null: same proportions
    total_obs = falls_obs + finds_obs
    expected_falls = total_obs * (falls_total / (falls_total + finds_total))
    expected_finds = total_obs * (finds_total / (falls_total + finds_total))

    chi2_falls = np.sum((falls_obs - expected_falls) ** 2 / expected_falls)
    chi2_finds = np.sum((finds_obs - expected_finds) ** 2 / expected_finds)
    chi2_total = chi2_falls + chi2_finds

    dof = len(valid_classes) - 1
    chi2_p = 1 - stats.chi2.cdf(chi2_total, dof)

    # Verdict
    n_significant = len(sig_classes)
    if chi2_p < 0.001 and n_significant >= 3:
        verdict = "STRONG_DIFFERENCE"
        interpretation = f"Falls and Finds have significantly different class compositions ({n_significant} classes differ)"
    elif chi2_p < 0.05:
        verdict = "MODERATE_DIFFERENCE"
        interpretation = "Some difference in class composition detected"
    else:
        verdict = "NO_DIFFERENCE"
        interpretation = "Falls and Finds have similar class proportions"

    return {
        "test": "h_fall_2",
        "status": "completed",
        "falls_total": int(falls_total),
        "finds_total": int(finds_total),
        "n_classes_compared": len(valid_classes),
        "chi2_statistic": float(chi2_total),
        "chi2_dof": dof,
        "chi2_p": float(chi2_p),
        "n_significant_classes": int(n_significant),
        "over_represented": over_represented[["class", "enrichment_ratio", "fisher_p"]].to_dict("records") if len(over_represented) > 0 else [],
        "under_represented": under_represented[["class", "enrichment_ratio", "fisher_p"]].to_dict("records") if len(under_represented) > 0 else [],
        "verdict": verdict,
        "interpretation": interpretation,
        "all_enrichment": enrich_df.to_dict("records"),
    }


def test_h_fall_3_seasonality(
    falls_df: pd.DataFrame,
    n_permutations: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    H-FALL-3: Test if Falls show seasonality.

    If meteorite flux varies with Earth's position in orbit,
    this would reveal something about the source populations.
    """
    np.random.seed(seed)

    # We need month data - extract from year if available or use recmonth
    # The NASA dataset may have 'recmonth' or we need to parse dates

    # Check if we have month information
    if "month" in falls_df.columns:
        month_col = "month"
    elif "recmonth" in falls_df.columns:
        month_col = "recmonth"
    else:
        # Try to extract from a date column or use year fractional
        return {
            "test": "h_fall_3",
            "status": "no_month_data",
            "message": "Month information not available in dataset",
        }

    df = falls_df.dropna(subset=[month_col]).copy()
    months = df[month_col].values

    # Convert to numeric if needed
    months = pd.to_numeric(months, errors='coerce')
    df = df[~np.isnan(months)]
    months = months[~np.isnan(months)].astype(int)

    if len(months) < 100:
        return {
            "test": "h_fall_3",
            "status": "insufficient_data",
            "message": f"Only {len(months)} falls with month data",
        }

    # Count by month
    month_counts = pd.Series(months).value_counts().sort_index()

    # Fill missing months with 0
    all_months = pd.Series(0, index=range(1, 13))
    for m, c in month_counts.items():
        if 1 <= m <= 12:
            all_months[m] = c

    observed_counts = all_months.values
    total = observed_counts.sum()

    # Expected under uniform: equal probability each month
    # But months have different lengths, so adjust
    days_per_month = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    total_days = sum(days_per_month)
    expected_counts = np.array([total * d / total_days for d in days_per_month])

    # Chi-square test
    chi2_stat = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
    chi2_p = 1 - stats.chi2.cdf(chi2_stat, 11)  # 12 months - 1 = 11 dof

    # Rayleigh test for circular uniformity (more powerful for seasonality)
    # Convert months to angles (radians)
    angles = 2 * np.pi * (months - 0.5) / 12

    # Mean resultant length
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R = np.sqrt(C**2 + S**2)
    n = len(angles)
    R_bar = R / n

    # Rayleigh test statistic
    rayleigh_z = n * R_bar ** 2
    rayleigh_p = np.exp(-rayleigh_z)  # Approximate p-value

    # Mean direction (peak month)
    mean_angle = np.arctan2(S, C)
    peak_month = (mean_angle * 12 / (2 * np.pi) + 0.5) % 12 + 1

    # Permutation test for robustness
    observed_R = R_bar
    null_Rs = []
    for _ in range(n_permutations):
        perm_months = np.random.randint(1, 13, size=n)
        perm_angles = 2 * np.pi * (perm_months - 0.5) / 12
        C_p = np.sum(np.cos(perm_angles))
        S_p = np.sum(np.sin(perm_angles))
        null_Rs.append(np.sqrt(C_p**2 + S_p**2) / n)

    perm_p = np.mean([nr >= observed_R for nr in null_Rs])

    # Identify peak and trough months
    peak_months = [m for m in range(1, 13) if observed_counts[m-1] > expected_counts[m-1] * 1.2]
    trough_months = [m for m in range(1, 13) if observed_counts[m-1] < expected_counts[m-1] * 0.8]

    # Verdict
    if rayleigh_p < 0.01 and perm_p < 0.05:
        verdict = "SEASONALITY_DETECTED"
        interpretation = f"Falls show significant seasonality (peak near month {peak_month:.1f})"
    elif rayleigh_p < 0.05 or perm_p < 0.10:
        verdict = "WEAK_SEASONALITY"
        interpretation = "Marginal evidence for seasonality in Falls"
    else:
        verdict = "NO_SEASONALITY"
        interpretation = "Falls are consistent with uniform distribution across months"

    return {
        "test": "h_fall_3",
        "status": "completed",
        "n_falls": int(n),
        "chi2_stat": float(chi2_stat),
        "chi2_p": float(chi2_p),
        "rayleigh_R_bar": float(R_bar),
        "rayleigh_p": float(rayleigh_p),
        "permutation_p": float(perm_p),
        "peak_month": float(peak_month),
        "peak_months": peak_months,
        "trough_months": trough_months,
        "month_counts": {int(m): int(c) for m, c in zip(range(1, 13), observed_counts)},
        "month_expected": {int(m): float(e) for m, e in zip(range(1, 13), expected_counts)},
        "verdict": verdict,
        "interpretation": interpretation,
    }


def test_mass_distribution_comparison(
    falls_df: pd.DataFrame,
    finds_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compare mass distributions between Falls and Finds.

    Falls should represent the true incoming flux.
    Finds are biased toward larger/more visible meteorites.
    """
    falls_masses = falls_df["mass"].dropna()
    finds_masses = finds_df["mass"].dropna()

    falls_masses = falls_masses[falls_masses > 0]
    finds_masses = finds_masses[finds_masses > 0]

    if len(falls_masses) < 50 or len(finds_masses) < 50:
        return {"test": "mass_comparison", "status": "insufficient_data"}

    # Log-transform for analysis
    log_falls = np.log(falls_masses)
    log_finds = np.log(finds_masses)

    # Basic statistics
    stats_comparison = {
        "falls": {
            "n": int(len(falls_masses)),
            "median": float(np.median(falls_masses)),
            "mean": float(np.mean(falls_masses)),
            "log_mean": float(np.mean(log_falls)),
            "log_std": float(np.std(log_falls)),
            "percentile_25": float(np.percentile(falls_masses, 25)),
            "percentile_75": float(np.percentile(falls_masses, 75)),
        },
        "finds": {
            "n": int(len(finds_masses)),
            "median": float(np.median(finds_masses)),
            "mean": float(np.mean(finds_masses)),
            "log_mean": float(np.mean(log_finds)),
            "log_std": float(np.std(log_finds)),
            "percentile_25": float(np.percentile(finds_masses, 25)),
            "percentile_75": float(np.percentile(finds_masses, 75)),
        },
    }

    # KS test
    ks_stat, ks_p = stats.ks_2samp(log_falls, log_finds)

    # Mann-Whitney U test
    mw_stat, mw_p = stats.mannwhitneyu(falls_masses, finds_masses, alternative='two-sided')

    # Effect size: ratio of medians
    median_ratio = np.median(falls_masses) / np.median(finds_masses)

    # Interpretation
    if ks_p < 0.001:
        verdict = "VERY_DIFFERENT"
        interpretation = f"Falls and Finds have very different mass distributions (Falls median {median_ratio:.1f}x Finds)"
    elif ks_p < 0.05:
        verdict = "DIFFERENT"
        interpretation = "Mass distributions differ significantly"
    else:
        verdict = "SIMILAR"
        interpretation = "Mass distributions are statistically similar"

    return {
        "test": "mass_comparison",
        "status": "completed",
        "stats": stats_comparison,
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "mannwhitney_stat": float(mw_stat),
        "mannwhitney_p": float(mw_p),
        "median_ratio_falls_over_finds": float(median_ratio),
        "verdict": verdict,
        "interpretation": interpretation,
    }


def run_o_delta_10() -> Dict[str, Any]:
    """
    Run O-Δ10: Falls-Only Analysis.
    """
    print("=" * 70)
    print("O-Δ10: Falls-Only Analysis — The Unbiased Sample")
    print("=" * 70)
    print()

    # Load data
    df = pd.read_parquet(DATA_PROCESSED / "meteorites.parquet")

    # Split into Falls and Finds
    falls_df = df[df["fall"] == "Fell"].copy()
    finds_df = df[df["fall"] == "Found"].copy()

    print(f"Total samples: {len(df):,}")
    print(f"Falls: {len(falls_df):,} ({100*len(falls_df)/len(df):.1f}%)")
    print(f"Finds: {len(finds_df):,} ({100*len(finds_df)/len(df):.1f}%)")
    print()

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1: Falls Profile
    # ════════════════════════════════════════════════════════════════════
    print("-" * 70)
    print("PHASE 1: Falls Profile")
    print("-" * 70)

    falls_profile = compute_falls_profile(falls_df)
    finds_profile = compute_falls_profile(finds_df)

    print(f"\n  FALLS:")
    print(f"    Total: {falls_profile.n_total:,}")
    print(f"    With mass: {falls_profile.n_with_mass:,}")
    print(f"    Classes: {falls_profile.n_classes}")
    print(f"    Median mass: {falls_profile.mass_median:,.0f}g")
    print(f"    Year range: {falls_profile.year_range}")
    print(f"    Top classes: {falls_profile.top_classes[:5]}")

    print(f"\n  FINDS (for comparison):")
    print(f"    Total: {finds_profile.n_total:,}")
    print(f"    Median mass: {finds_profile.mass_median:,.0f}g")
    print(f"    Classes: {finds_profile.n_classes}")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2: Mass Distribution Comparison
    # ════════════════════════════════════════════════════════════════════
    print()
    print("-" * 70)
    print("PHASE 2: Mass Distribution Comparison")
    print("-" * 70)

    mass_comparison = test_mass_distribution_comparison(falls_df, finds_df)
    print(f"\n  Falls median: {mass_comparison['stats']['falls']['median']:,.0f}g")
    print(f"  Finds median: {mass_comparison['stats']['finds']['median']:,.0f}g")
    print(f"  Ratio: {mass_comparison['median_ratio_falls_over_finds']:.1f}x")
    print(f"  KS p-value: {mass_comparison['ks_p']:.2e}")
    print(f"  → {mass_comparison['verdict']}")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 3: H-FALL-1 — Mass Structure in Falls
    # ════════════════════════════════════════════════════════════════════
    print()
    print("-" * 70)
    print("PHASE 3: H-FALL-1 — Genuine Mass Structure in Falls")
    print("-" * 70)

    h_fall_1 = test_h_fall_1_mass_structure(falls_df, n_permutations=1000, subsample_size=20)

    if h_fall_1["status"] == "completed":
        print(f"\n  Classes analyzed: {h_fall_1['n_classes']}")
        print(f"  Subsample size: {h_fall_1['subsample_size']}")
        print(f"  Observed CV spread: {h_fall_1['observed_cv_spread']:.4f}")
        print(f"  Null CV spread: {h_fall_1['null_cv_spread_mean']:.4f} ± {h_fall_1['null_cv_spread_std']:.4f}")
        print(f"  p-value: {h_fall_1['p_value']:.4f}")
        print(f"  → {h_fall_1['verdict']}")
    else:
        print(f"  Status: {h_fall_1['status']}")
        print(f"  Message: {h_fall_1.get('message', 'N/A')}")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 4: H-FALL-2 — Class Proportion Differences
    # ════════════════════════════════════════════════════════════════════
    print()
    print("-" * 70)
    print("PHASE 4: H-FALL-2 — Class Proportions Falls vs Finds")
    print("-" * 70)

    h_fall_2 = test_h_fall_2_class_proportions(falls_df, finds_df)

    if h_fall_2["status"] == "completed":
        print(f"\n  Classes compared: {h_fall_2['n_classes_compared']}")
        print(f"  Chi-square: {h_fall_2['chi2_statistic']:.1f} (p={h_fall_2['chi2_p']:.2e})")
        print(f"  Significant classes: {h_fall_2['n_significant_classes']}")

        if h_fall_2["over_represented"]:
            print(f"\n  OVER-REPRESENTED in Falls (vs Finds):")
            for item in h_fall_2["over_represented"][:5]:
                print(f"    {item['class']}: {item['enrichment_ratio']:.2f}x (p={item['fisher_p']:.4f})")

        if h_fall_2["under_represented"]:
            print(f"\n  UNDER-REPRESENTED in Falls (vs Finds):")
            for item in h_fall_2["under_represented"][:5]:
                print(f"    {item['class']}: {item['enrichment_ratio']:.2f}x (p={item['fisher_p']:.4f})")

        print(f"\n  → {h_fall_2['verdict']}")
    else:
        print(f"  Status: {h_fall_2['status']}")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 5: H-FALL-3 — Seasonality
    # ════════════════════════════════════════════════════════════════════
    print()
    print("-" * 70)
    print("PHASE 5: H-FALL-3 — Seasonality in Falls")
    print("-" * 70)

    h_fall_3 = test_h_fall_3_seasonality(falls_df)

    if h_fall_3["status"] == "completed":
        print(f"\n  Falls with month data: {h_fall_3['n_falls']}")
        print(f"  Rayleigh R: {h_fall_3['rayleigh_R_bar']:.4f}")
        print(f"  Rayleigh p: {h_fall_3['rayleigh_p']:.4f}")
        print(f"  Permutation p: {h_fall_3['permutation_p']:.4f}")
        print(f"  Peak month: {h_fall_3['peak_month']:.1f}")
        print(f"  → {h_fall_3['verdict']}")
    else:
        print(f"  Status: {h_fall_3['status']}")
        print(f"  Message: {h_fall_3.get('message', 'N/A')}")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "total": len(df),
            "falls": len(falls_df),
            "finds": len(finds_df),
            "falls_pct": float(100 * len(falls_df) / len(df)),
        },
        "falls_profile": {
            "n_total": falls_profile.n_total,
            "n_with_mass": falls_profile.n_with_mass,
            "n_classes": falls_profile.n_classes,
            "mass_median": falls_profile.mass_median,
            "mass_log_mean": falls_profile.mass_log_mean,
            "mass_log_std": falls_profile.mass_log_std,
            "year_range": list(falls_profile.year_range),
            "top_classes": falls_profile.top_classes,
        },
        "mass_comparison": mass_comparison,
        "h_fall_1": h_fall_1,
        "h_fall_2": h_fall_2,
        "h_fall_3": h_fall_3,
    }


def generate_falls_plots(results: Dict[str, Any], output_path: Path):
    """Generate O-Δ10 plots."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ─────────────────────────────────────────────────────────────────
    # Plot 1: Falls vs Finds counts
    # ─────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    categories = ["Falls", "Finds"]
    counts = [results["dataset"]["falls"], results["dataset"]["finds"]]
    colors = ["#4169E1", "#CD853F"]

    bars = ax1.bar(categories, counts, color=colors)
    ax1.set_ylabel("Number of Samples")
    ax1.set_title("Falls vs Finds in Catalog")

    for bar, n in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{n:,}', ha='center', va='bottom', fontsize=11)

    ax1.set_yscale('log')

    # ─────────────────────────────────────────────────────────────────
    # Plot 2: Mass distribution comparison
    # ─────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    mc = results["mass_comparison"]
    if mc["status"] == "completed":
        categories = ["Falls", "Finds"]
        medians = [mc["stats"]["falls"]["median"], mc["stats"]["finds"]["median"]]

        bars = ax2.bar(categories, medians, color=colors)
        ax2.set_ylabel("Median Mass (g)")
        ax2.set_title(f"Mass Comparison\n{mc['verdict']} (p={mc['ks_p']:.2e})")
        ax2.set_yscale('log')

        for bar, m in zip(bars, medians):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{m:,.0f}g', ha='center', va='bottom', fontsize=10)

    # ─────────────────────────────────────────────────────────────────
    # Plot 3: Class enrichment
    # ─────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    h2 = results["h_fall_2"]
    if h2["status"] == "completed" and h2["all_enrichment"]:
        enrich_df = pd.DataFrame(h2["all_enrichment"])
        enrich_df = enrich_df.sort_values("enrichment_ratio", ascending=True)

        # Top 10 over and under represented
        top_over = enrich_df.tail(10)
        top_under = enrich_df.head(10)
        combined = pd.concat([top_under, top_over])

        colors_enrich = ['#228B22' if r > 1 else '#DC143C' for r in combined["enrichment_ratio"]]

        ax3.barh(range(len(combined)), combined["enrichment_ratio"], color=colors_enrich)
        ax3.set_yticks(range(len(combined)))
        ax3.set_yticklabels(combined["class"], fontsize=8)
        ax3.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel("Enrichment Ratio (Falls/Finds)")
        ax3.set_title(f"Class Enrichment in Falls\n(Green=over, Red=under)")
        ax3.set_xscale('log')

    # ─────────────────────────────────────────────────────────────────
    # Plot 4: Seasonality (if available)
    # ─────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    h3 = results["h_fall_3"]
    if h3["status"] == "completed":
        months = list(range(1, 13))
        observed = [h3["month_counts"].get(m, 0) for m in months]
        expected = [h3["month_expected"].get(m, 0) for m in months]

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        x = np.arange(12)
        width = 0.35

        ax4.bar(x - width/2, observed, width, label='Observed', color='#4169E1')
        ax4.bar(x + width/2, expected, width, label='Expected', color='#CD853F', alpha=0.7)

        ax4.set_xticks(x)
        ax4.set_xticklabels(month_names, rotation=45)
        ax4.set_ylabel("Number of Falls")
        ax4.set_title(f"Seasonality: {h3['verdict']}\n(Rayleigh p={h3['rayleigh_p']:.4f})")
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, f"Seasonality test:\n{h3.get('status', 'N/A')}\n{h3.get('message', '')}",
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("Seasonality Analysis")

    plt.suptitle("O-Δ10: Falls-Only Analysis — The Unbiased Sample",
                fontsize=14, fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_outputs(results: Dict[str, Any], output_dir: Path = REPORTS) -> Dict[str, str]:
    """Generate O-Δ10 outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    # 1. JSON results
    json_path = output_dir / "O-D10_falls_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    files["json"] = str(json_path)

    # 2. Hypothesis summary CSV
    hyp_data = []

    h1 = results["h_fall_1"]
    hyp_data.append({
        "hypothesis": "H-FALL-1",
        "description": "Genuine mass structure in Falls",
        "status": h1.get("status", "N/A"),
        "verdict": h1.get("verdict", "N/A"),
        "p_value": h1.get("p_value", np.nan),
    })

    h2 = results["h_fall_2"]
    hyp_data.append({
        "hypothesis": "H-FALL-2",
        "description": "Class proportions differ Falls vs Finds",
        "status": h2.get("status", "N/A"),
        "verdict": h2.get("verdict", "N/A"),
        "p_value": h2.get("chi2_p", np.nan),
    })

    h3 = results["h_fall_3"]
    hyp_data.append({
        "hypothesis": "H-FALL-3",
        "description": "Seasonality in Falls",
        "status": h3.get("status", "N/A"),
        "verdict": h3.get("verdict", "N/A"),
        "p_value": h3.get("rayleigh_p", np.nan),
    })

    hyp_df = pd.DataFrame(hyp_data)
    hyp_path = output_dir / "O-D10_hypothesis_summary.csv"
    hyp_df.to_csv(hyp_path, index=False)
    files["hypothesis_csv"] = str(hyp_path)

    # 3. Enrichment CSV
    h2 = results["h_fall_2"]
    if h2.get("status") == "completed" and h2.get("all_enrichment"):
        enrich_df = pd.DataFrame(h2["all_enrichment"])
        enrich_path = output_dir / "O-D10_class_enrichment.csv"
        enrich_df.to_csv(enrich_path, index=False)
        files["enrichment_csv"] = str(enrich_path)

    # 4. Plot
    plot_path = output_dir / "O-D10_falls_analysis.png"
    generate_falls_plots(results, plot_path)
    files["plot"] = str(plot_path)

    print()
    print("=" * 70)
    print("FILES GENERATED")
    print("=" * 70)
    for k, v in files.items():
        print(f"  {k}: {v}")

    return files


def generate_observation_md(results: Dict[str, Any], files: Dict[str, str], output_dir: Path) -> str:
    """Generate observation markdown."""
    date = datetime.now().strftime("%Y%m%d")
    md_path = output_dir / f"observation_O-D10_{date}.md"

    h1 = results["h_fall_1"]
    h2 = results["h_fall_2"]
    h3 = results["h_fall_3"]
    mc = results["mass_comparison"]

    lines = [
        "# Observation O-Δ10: Falls-Only Analysis",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Experiment**: O-Δ10 (The Unbiased Sample)",
        "",
        "---",
        "",
        "## Rationale",
        "",
        "Falls (meteorites we saw fall) have **no collection bias**:",
        "- No size bias — we didn't search for them",
        "- No visibility bias — we saw them fall",
        "- No location bias — they fell where they fell",
        "",
        "This is our best window into the TRUE flux of meteorites to Earth.",
        "",
        "---",
        "",
        "## Dataset",
        "",
        f"| Category | N | % |",
        f"|----------|---|---|",
        f"| Falls | {results['dataset']['falls']:,} | {results['dataset']['falls_pct']:.1f}% |",
        f"| Finds | {results['dataset']['finds']:,} | {100-results['dataset']['falls_pct']:.1f}% |",
        "",
        "---",
        "",
        "## Mass Comparison: Falls vs Finds",
        "",
    ]

    if mc["status"] == "completed":
        lines.extend([
            f"| Metric | Falls | Finds |",
            f"|--------|-------|-------|",
            f"| Median mass | {mc['stats']['falls']['median']:,.0f}g | {mc['stats']['finds']['median']:,.0f}g |",
            f"| Mean mass | {mc['stats']['falls']['mean']:,.0f}g | {mc['stats']['finds']['mean']:,.0f}g |",
            "",
            f"**Ratio**: Falls are {mc['median_ratio_falls_over_finds']:.1f}x larger than Finds",
            f"**KS test**: p = {mc['ks_p']:.2e}",
            f"**Verdict**: {mc['verdict']}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Hypothesis Results",
        "",
        "| Hypothesis | Description | Verdict | p-value |",
        "|------------|-------------|---------|---------|",
    ])

    h1_p = f"{h1.get('p_value', 'N/A'):.4f}" if isinstance(h1.get('p_value'), float) else "N/A"
    h2_p = f"{h2.get('chi2_p', 'N/A'):.2e}" if isinstance(h2.get('chi2_p'), float) else "N/A"
    h3_p = f"{h3.get('rayleigh_p', 'N/A'):.4f}" if isinstance(h3.get('rayleigh_p'), float) else "N/A"

    lines.extend([
        f"| H-FALL-1 | Mass structure in Falls | {h1.get('verdict', h1.get('status', 'N/A'))} | {h1_p} |",
        f"| H-FALL-2 | Class proportions differ | {h2.get('verdict', h2.get('status', 'N/A'))} | {h2_p} |",
        f"| H-FALL-3 | Seasonality in Falls | {h3.get('verdict', h3.get('status', 'N/A'))} | {h3_p} |",
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ])

    # Add key findings based on results
    findings = []

    if mc["status"] == "completed" and mc["verdict"] in ["VERY_DIFFERENT", "DIFFERENT"]:
        findings.append(f"1. **Falls are much larger than Finds** — {mc['median_ratio_falls_over_finds']:.0f}x median mass difference confirms collection bias in Finds")

    if h1.get("verdict") == "STRUCTURE_DETECTED":
        findings.append("2. **GENUINE mass structure exists in Falls** — This survives balanced subsampling!")
    elif h1.get("verdict") == "NO_STRUCTURE":
        findings.append("2. **No genuine mass structure in Falls** — Even in the unbiased sample, class heterogeneity is random")

    if h2.get("verdict") in ["STRONG_DIFFERENCE", "MODERATE_DIFFERENCE"]:
        n_sig = h2.get("n_significant_classes", 0)
        findings.append(f"3. **Class proportions differ between Falls and Finds** — {n_sig} classes significantly enriched/depleted")

        if h2.get("over_represented"):
            over = [x["class"] for x in h2["over_represented"][:3]]
            findings.append(f"   - Over-represented in Falls: {', '.join(over)}")

        if h2.get("under_represented"):
            under = [x["class"] for x in h2["under_represented"][:3]]
            findings.append(f"   - Under-represented in Falls: {', '.join(under)}")

    if h3.get("verdict") == "SEASONALITY_DETECTED":
        findings.append(f"4. **Seasonality detected** — Falls peak around month {h3.get('peak_month', 'N/A'):.0f}")
    elif h3.get("verdict") == "NO_SEASONALITY":
        findings.append("4. **No seasonality** — Falls are uniformly distributed across months")

    lines.extend(findings)

    lines.extend([
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ])

    # Overall interpretation
    n_positive = sum(1 for v in [
        h1.get("verdict") == "STRUCTURE_DETECTED",
        h2.get("verdict") in ["STRONG_DIFFERENCE", "MODERATE_DIFFERENCE"],
        h3.get("verdict") == "SEASONALITY_DETECTED",
    ] if v)

    if n_positive >= 2:
        lines.append("**Falls reveal genuine physical patterns** that are masked or distorted in the full catalog.")
        lines.append("The unbiased sample shows real structure in the meteorite flux.")
    elif n_positive == 1:
        lines.append("**Mixed results**: Some patterns in Falls, but not conclusive evidence of genuine physical structure.")
    else:
        lines.append("**Falls confirm the null**: Even in the unbiased sample, we find no strong evidence of genuine physical patterns.")
        lines.append("The meteorite flux may be truly random with respect to these properties.")

    lines.extend([
        "",
        "---",
        "",
        "## Files Generated",
        "",
    ])

    for k, v in files.items():
        lines.append(f"- `{k}`: {v}")

    lines.extend([
        "",
        "---",
        "",
        "*Generated by ORIGINMAP O-Δ10 experiment*",
    ])

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return str(md_path)


def run_full_o_delta_10() -> Dict[str, str]:
    """Main entry point for O-Δ10."""
    results = run_o_delta_10()
    files = generate_outputs(results, REPORTS)

    # Generate observation
    from originmap.config import PROJECT_ROOT
    obs_dir = PROJECT_ROOT / "notes" / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)
    obs_path = generate_observation_md(results, files, obs_dir)
    files["observation"] = obs_path

    # Print verdict
    print()
    print("=" * 70)
    print("O-Δ10 VERDICT")
    print("=" * 70)

    h1 = results["h_fall_1"]
    h2 = results["h_fall_2"]
    h3 = results["h_fall_3"]

    print(f"\n  H-FALL-1 (mass structure):  {h1.get('verdict', h1.get('status', 'N/A'))}")
    print(f"  H-FALL-2 (class diff):       {h2.get('verdict', h2.get('status', 'N/A'))}")
    print(f"  H-FALL-3 (seasonality):      {h3.get('verdict', h3.get('status', 'N/A'))}")

    # Count discoveries
    discoveries = []
    if h1.get("verdict") == "STRUCTURE_DETECTED":
        discoveries.append("genuine mass structure")
    if h2.get("verdict") in ["STRONG_DIFFERENCE", "MODERATE_DIFFERENCE"]:
        discoveries.append("class composition differences")
    if h3.get("verdict") == "SEASONALITY_DETECTED":
        discoveries.append("seasonality")

    print()
    if discoveries:
        print(f"  → DISCOVERIES: {', '.join(discoveries)}")
    else:
        print("  → No strong patterns detected in Falls")

    return files


if __name__ == "__main__":
    run_full_o_delta_10()
