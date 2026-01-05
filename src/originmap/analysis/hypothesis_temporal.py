"""
O-Δ9: Catalog Archaeology — Temporal Dynamics of Meteorite Discovery

Key question: Does the meteorite catalog reflect the universe,
or does it reflect the history of human observation?

Hypotheses:
- H-TEMP-1: Classes appear in "waves" (not uniformly)
- H-TEMP-2: Mean discovery mass DECREASES over time (large ones found first)
- H-TEMP-3: Antarctica distorts the catalog post-1970
- H-TEMP-4: There are "fashions" in classification (classifier bias)

Eras defined:
- Pre-1900: Classical era
- 1900-1969: Modern pre-Antarctica
- 1970-1999: Antarctica boom
- 2000+: Satellite/drone era
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


# Era definitions
ERAS = {
    "classical": (0, 1899),
    "modern_pre_antarctica": (1900, 1969),
    "antarctica_boom": (1970, 1999),
    "satellite_era": (2000, 2100),
}

ERA_COLORS = {
    "classical": "#8B4513",
    "modern_pre_antarctica": "#4169E1",
    "antarctica_boom": "#228B22",
    "satellite_era": "#FF6347",
}


def assign_era(year: float) -> Optional[str]:
    """Assign a year to an era."""
    if pd.isna(year):
        return None
    year = int(year)
    for era_name, (start, end) in ERAS.items():
        if start <= year <= end:
            return era_name
    return None


def shannon_diversity(counts: np.ndarray) -> float:
    """Calculate Shannon diversity index."""
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    proportions = counts / counts.sum()
    return -np.sum(proportions * np.log(proportions))


def simpson_diversity(counts: np.ndarray) -> float:
    """Calculate Simpson diversity index (1 - D)."""
    counts = counts[counts > 0]
    if len(counts) == 0 or counts.sum() < 2:
        return 0.0
    n = counts.sum()
    return 1 - np.sum(counts * (counts - 1)) / (n * (n - 1))


@dataclass
class EraMetrics:
    """Metrics for a single era."""
    era_name: str
    year_range: Tuple[int, int]
    n_total: int
    n_falls: int
    n_finds: int
    fall_ratio: float
    n_classes: int
    shannon_diversity: float
    simpson_diversity: float
    new_classes: List[str]  # Classes first seen in this era
    n_new_classes: int
    mass_median: float
    mass_mean: float
    mass_std: float
    mass_log_mean: float
    mass_log_std: float
    top_classes: List[Tuple[str, int]]  # Top 10 classes by count

    def to_dict(self) -> Dict:
        return {
            "era_name": self.era_name,
            "year_range": list(self.year_range),
            "n_total": self.n_total,
            "n_falls": self.n_falls,
            "n_finds": self.n_finds,
            "fall_ratio": self.fall_ratio,
            "n_classes": self.n_classes,
            "shannon_diversity": self.shannon_diversity,
            "simpson_diversity": self.simpson_diversity,
            "new_classes": self.new_classes,
            "n_new_classes": self.n_new_classes,
            "mass_median": self.mass_median,
            "mass_mean": self.mass_mean,
            "mass_std": self.mass_std,
            "mass_log_mean": self.mass_log_mean,
            "mass_log_std": self.mass_log_std,
            "top_classes": self.top_classes,
        }


def compute_era_metrics(
    df: pd.DataFrame,
    era_name: str,
    previous_classes: set
) -> Tuple[EraMetrics, set]:
    """
    Compute metrics for a single era.

    Returns metrics and updated set of all classes seen so far.
    """
    year_range = ERAS[era_name]

    # Filter to era
    era_df = df[df["era"] == era_name]

    n_total = len(era_df)

    if n_total == 0:
        return EraMetrics(
            era_name=era_name,
            year_range=year_range,
            n_total=0,
            n_falls=0, n_finds=0, fall_ratio=0,
            n_classes=0, shannon_diversity=0, simpson_diversity=0,
            new_classes=[], n_new_classes=0,
            mass_median=np.nan, mass_mean=np.nan, mass_std=np.nan,
            mass_log_mean=np.nan, mass_log_std=np.nan,
            top_classes=[]
        ), previous_classes

    # Fall/Find
    fall_counts = era_df["fall"].fillna("Unknown").value_counts()
    n_falls = fall_counts.get("Fell", 0)
    n_finds = fall_counts.get("Found", 0)
    fall_ratio = n_falls / n_total if n_total > 0 else 0

    # Class diversity
    class_counts = era_df["recclass"].value_counts()
    n_classes = len(class_counts)
    counts_array = class_counts.values
    shannon = shannon_diversity(counts_array)
    simpson = simpson_diversity(counts_array)

    # New classes
    current_classes = set(class_counts.index)
    new_classes = list(current_classes - previous_classes)
    all_classes = previous_classes | current_classes

    # Mass statistics
    masses = era_df["mass"].dropna()
    masses = masses[masses > 0]

    if len(masses) > 0:
        mass_median = np.median(masses)
        mass_mean = np.mean(masses)
        mass_std = np.std(masses)
        log_masses = np.log(masses)
        mass_log_mean = np.mean(log_masses)
        mass_log_std = np.std(log_masses)
    else:
        mass_median = mass_mean = mass_std = np.nan
        mass_log_mean = mass_log_std = np.nan

    # Top classes
    top_classes = [(c, int(n)) for c, n in class_counts.head(10).items()]

    return EraMetrics(
        era_name=era_name,
        year_range=year_range,
        n_total=n_total,
        n_falls=n_falls,
        n_finds=n_finds,
        fall_ratio=fall_ratio,
        n_classes=n_classes,
        shannon_diversity=shannon,
        simpson_diversity=simpson,
        new_classes=new_classes,
        n_new_classes=len(new_classes),
        mass_median=mass_median,
        mass_mean=mass_mean,
        mass_std=mass_std,
        mass_log_mean=mass_log_mean,
        mass_log_std=mass_log_std,
        top_classes=top_classes,
    ), all_classes


def detect_changepoints_mass(
    df: pd.DataFrame,
    window_years: int = 10
) -> List[Dict]:
    """
    Detect change points in mass distribution over time.
    Uses rolling median and detects significant shifts.
    """
    # Group by decade
    df_valid = df.dropna(subset=["year", "mass"])
    df_valid = df_valid[df_valid["mass"] > 0].copy()
    df_valid["decade"] = (df_valid["year"] // window_years) * window_years

    decade_stats = df_valid.groupby("decade").agg({
        "mass": ["median", "mean", "count"],
    }).reset_index()
    decade_stats.columns = ["decade", "median_mass", "mean_mass", "count"]
    decade_stats = decade_stats[decade_stats["count"] >= 10]

    if len(decade_stats) < 3:
        return []

    # Detect change points using log-median
    log_medians = np.log(decade_stats["median_mass"].values)
    decades = decade_stats["decade"].values

    changepoints = []

    # Simple change point: where derivative changes sign significantly
    if len(log_medians) >= 5:
        diffs = np.diff(log_medians)

        for i in range(1, len(diffs)):
            # Sign change with magnitude
            if diffs[i-1] * diffs[i] < 0 and abs(diffs[i] - diffs[i-1]) > 0.3:
                changepoints.append({
                    "decade": int(decades[i]),
                    "type": "trend_reversal",
                    "before_median": float(np.exp(log_medians[i-1])),
                    "after_median": float(np.exp(log_medians[i])),
                    "magnitude": float(abs(diffs[i] - diffs[i-1])),
                })

    return changepoints


def detect_changepoints_diversity(
    df: pd.DataFrame,
    window_years: int = 10
) -> List[Dict]:
    """
    Detect change points in class diversity over time.
    """
    df_valid = df.dropna(subset=["year", "recclass"]).copy()
    df_valid["decade"] = (df_valid["year"] // window_years) * window_years

    decade_diversity = []

    for decade, group in df_valid.groupby("decade"):
        if len(group) >= 10:
            class_counts = group["recclass"].value_counts().values
            shannon = shannon_diversity(class_counts)
            decade_diversity.append({
                "decade": decade,
                "shannon": shannon,
                "n_classes": len(class_counts),
                "n_samples": len(group),
            })

    if len(decade_diversity) < 3:
        return []

    div_df = pd.DataFrame(decade_diversity)

    changepoints = []

    # Detect sudden jumps in diversity
    shannons = div_df["shannon"].values
    decades = div_df["decade"].values

    for i in range(1, len(shannons)):
        jump = shannons[i] - shannons[i-1]
        if abs(jump) > 0.5:  # Significant jump
            changepoints.append({
                "decade": int(decades[i]),
                "type": "diversity_jump",
                "before_shannon": float(shannons[i-1]),
                "after_shannon": float(shannons[i]),
                "jump": float(jump),
            })

    return changepoints


def test_mass_trend(df: pd.DataFrame) -> Dict:
    """
    Test H-TEMP-2: Mean discovery mass decreases over time.
    Uses Mann-Kendall trend test on decadal medians.
    """
    df_valid = df.dropna(subset=["year", "mass"])
    df_valid = df_valid[df_valid["mass"] > 0].copy()
    df_valid["decade"] = (df_valid["year"] // 10) * 10

    decade_medians = df_valid.groupby("decade")["mass"].median()
    decade_medians = decade_medians[decade_medians.index >= 1800]  # Valid data

    if len(decade_medians) < 5:
        return {"test": "insufficient_data"}

    decades = decade_medians.index.values
    medians = decade_medians.values
    log_medians = np.log(medians)

    # Spearman correlation (robust trend)
    spearman_r, spearman_p = stats.spearmanr(decades, log_medians)

    # Linear regression on log-medians
    slope, intercept, r_value, p_value, std_err = stats.linregress(decades, log_medians)

    # Interpretation
    if spearman_p < 0.05 and spearman_r < 0:
        trend = "DECREASING"
        h_temp_2 = "SUPPORTED"
    elif spearman_p < 0.05 and spearman_r > 0:
        trend = "INCREASING"
        h_temp_2 = "REJECTED"
    else:
        trend = "NO_TREND"
        h_temp_2 = "INCONCLUSIVE"

    return {
        "test": "mass_trend",
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "slope_per_decade": float(slope),
        "r_squared": float(r_value ** 2),
        "trend": trend,
        "h_temp_2": h_temp_2,
        "decades": decades.tolist(),
        "log_medians": log_medians.tolist(),
    }


def test_diversity_growth(df: pd.DataFrame) -> Dict:
    """
    Test if diversity grows with time (discovery vs real).
    """
    df_valid = df.dropna(subset=["year", "recclass"]).copy()
    df_valid["decade"] = (df_valid["year"] // 10) * 10

    cumulative_classes = []
    all_classes = set()

    for decade in sorted(df_valid["decade"].unique()):
        decade_classes = set(df_valid[df_valid["decade"] == decade]["recclass"])
        all_classes = all_classes | decade_classes
        cumulative_classes.append({
            "decade": decade,
            "cumulative_n_classes": len(all_classes),
            "new_in_decade": len(decade_classes - (all_classes - decade_classes)),
        })

    if len(cumulative_classes) < 5:
        return {"test": "insufficient_data"}

    cum_df = pd.DataFrame(cumulative_classes)
    cum_df = cum_df[cum_df["decade"] >= 1800]

    decades = cum_df["decade"].values
    n_classes = cum_df["cumulative_n_classes"].values

    # Test for saturation vs continued growth
    # Fit linear and log models

    # Linear
    slope_lin, intercept_lin, r_lin, p_lin, _ = stats.linregress(decades, n_classes)

    # Log (saturation model)
    log_decades = np.log(decades - decades.min() + 1)
    slope_log, intercept_log, r_log, p_log, _ = stats.linregress(log_decades, n_classes)

    if r_lin ** 2 > r_log ** 2:
        pattern = "LINEAR_GROWTH"
        interpretation = "Discovery continues at constant rate"
    else:
        pattern = "SATURATING"
        interpretation = "Discovery rate slowing (approaching true diversity?)"

    return {
        "test": "diversity_growth",
        "r_squared_linear": float(r_lin ** 2),
        "r_squared_log": float(r_log ** 2),
        "pattern": pattern,
        "interpretation": interpretation,
        "decades": decades.tolist(),
        "cumulative_classes": n_classes.tolist(),
        "slope_per_decade": float(slope_lin),
    }


def test_antarctica_effect(df: pd.DataFrame) -> Dict:
    """
    Test H-TEMP-3: Antarctica distorts the catalog post-1970.
    Compare pre-1970 vs post-1970 distributions.
    """
    df_valid = df.dropna(subset=["year", "mass", "recclass"])

    pre_1970 = df_valid[df_valid["year"] < 1970]
    post_1970 = df_valid[df_valid["year"] >= 1970]

    if len(pre_1970) < 100 or len(post_1970) < 100:
        return {"test": "insufficient_data"}

    # Compare mass distributions
    pre_masses = pre_1970["mass"].values
    post_masses = post_1970["mass"].values

    pre_masses = pre_masses[pre_masses > 0]
    post_masses = post_masses[post_masses > 0]

    # KS test on log-masses
    ks_stat, ks_p = stats.ks_2samp(np.log(pre_masses), np.log(post_masses))

    # Compare class distributions
    pre_classes = pre_1970["recclass"].value_counts()
    post_classes = post_1970["recclass"].value_counts()

    # Find classes that exploded post-1970
    ratio_changes = []
    all_classes = set(pre_classes.index) | set(post_classes.index)

    for c in all_classes:
        pre_n = pre_classes.get(c, 0)
        post_n = post_classes.get(c, 0)

        pre_frac = pre_n / len(pre_1970)
        post_frac = post_n / len(post_1970)

        if pre_frac > 0:
            ratio = post_frac / pre_frac
        elif post_frac > 0:
            ratio = float('inf')
        else:
            ratio = 1.0

        ratio_changes.append({
            "class": c,
            "pre_1970_n": int(pre_n),
            "post_1970_n": int(post_n),
            "pre_1970_frac": float(pre_frac),
            "post_1970_frac": float(post_frac),
            "ratio": float(ratio) if ratio != float('inf') else 999,
        })

    ratio_df = pd.DataFrame(ratio_changes)
    ratio_df = ratio_df.sort_values("ratio", ascending=False)

    # Top gainers and losers
    top_gainers = ratio_df.head(10).to_dict("records")
    top_losers = ratio_df.tail(10).to_dict("records")

    return {
        "test": "antarctica_effect",
        "pre_1970_n": int(len(pre_1970)),
        "post_1970_n": int(len(post_1970)),
        "mass_ks_stat": float(ks_stat),
        "mass_ks_p": float(ks_p),
        "mass_shift": "SIGNIFICANT" if ks_p < 0.05 else "NOT_SIGNIFICANT",
        "pre_1970_median_mass": float(np.median(pre_masses)),
        "post_1970_median_mass": float(np.median(post_masses)),
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "h_temp_3": "SUPPORTED" if ks_p < 0.001 else "WEAK" if ks_p < 0.05 else "REJECTED",
    }


def test_classification_waves(df: pd.DataFrame) -> Dict:
    """
    Test H-TEMP-1: Classes appear in waves.
    Look for burst patterns in class discovery.
    """
    df_valid = df.dropna(subset=["year", "recclass"]).copy()

    # For each class, find first and peak year
    class_temporal = []

    for recclass, group in df_valid.groupby("recclass"):
        if len(group) >= 5:
            years = group["year"].values
            first_year = int(np.min(years))
            peak_year = int(group.groupby("year").size().idxmax())
            last_year = int(np.max(years))
            span = last_year - first_year

            # Burstiness: is discovery concentrated or spread?
            year_counts = group["year"].value_counts()
            max_year_count = year_counts.max()
            total_count = len(group)
            concentration = max_year_count / total_count

            class_temporal.append({
                "recclass": recclass,
                "first_year": first_year,
                "peak_year": peak_year,
                "last_year": last_year,
                "span": span,
                "total": total_count,
                "concentration": concentration,
            })

    if len(class_temporal) < 10:
        return {"test": "insufficient_data"}

    ct_df = pd.DataFrame(class_temporal)

    # Test: are first_years clustered?
    first_years = ct_df["first_year"].values

    # Coefficient of variation of first years
    cv_first_years = np.std(first_years) / np.mean(first_years)

    # Test uniformity
    # If classes appear uniformly, first_years should be uniform
    # Use KS test against uniform distribution
    first_years_norm = (first_years - first_years.min()) / (first_years.max() - first_years.min())
    ks_stat, ks_p = stats.kstest(first_years_norm, 'uniform')

    # Find "wave" decades (many new classes)
    ct_df["first_decade"] = (ct_df["first_year"] // 10) * 10
    decade_counts = ct_df["first_decade"].value_counts().sort_index()

    wave_decades = decade_counts[decade_counts > decade_counts.median() * 2].index.tolist()

    return {
        "test": "classification_waves",
        "n_classes_analyzed": len(ct_df),
        "ks_uniform_stat": float(ks_stat),
        "ks_uniform_p": float(ks_p),
        "distribution": "NON_UNIFORM" if ks_p < 0.05 else "UNIFORM",
        "h_temp_1": "SUPPORTED" if ks_p < 0.05 else "REJECTED",
        "wave_decades": [int(d) for d in wave_decades],
        "new_classes_by_decade": {int(k): int(v) for k, v in decade_counts.items()},
        "high_concentration_classes": ct_df.nlargest(10, "concentration")[["recclass", "concentration", "first_year"]].to_dict("records"),
    }


def run_o_delta_9() -> Dict[str, Any]:
    """
    Run O-Δ9: Catalog Archaeology experiment.
    """
    print("=" * 70)
    print("O-Δ9: Catalog Archaeology — Temporal Dynamics")
    print("=" * 70)
    print()

    # Load data
    df = pd.read_parquet(DATA_PROCESSED / "meteorites.parquet")
    df = df.dropna(subset=["recclass"])

    # Assign eras
    df["era"] = df["year"].apply(assign_era)

    print(f"Dataset: {len(df)} samples")
    print(f"Year range: {df['year'].min():.0f} - {df['year'].max():.0f}")
    print(f"Valid years: {df['year'].notna().sum()}")
    print()

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1: Era Metrics
    # ════════════════════════════════════════════════════════════════════
    print("-" * 70)
    print("PHASE 1: Era Metrics")
    print("-" * 70)

    era_metrics = {}
    previous_classes = set()

    for era_name in ["classical", "modern_pre_antarctica", "antarctica_boom", "satellite_era"]:
        metrics, previous_classes = compute_era_metrics(df, era_name, previous_classes)
        era_metrics[era_name] = metrics

        print(f"\n  [{era_name.upper()}] {metrics.year_range}")
        print(f"    Samples: {metrics.n_total:,}")
        print(f"    Falls/Finds: {metrics.n_falls}/{metrics.n_finds} ({metrics.fall_ratio:.1%} falls)")
        print(f"    Classes: {metrics.n_classes} (NEW: {metrics.n_new_classes})")
        print(f"    Shannon diversity: {metrics.shannon_diversity:.2f}")
        print(f"    Median mass: {metrics.mass_median:,.0f}g")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2: Hypothesis Tests
    # ════════════════════════════════════════════════════════════════════
    print()
    print("-" * 70)
    print("PHASE 2: Hypothesis Tests")
    print("-" * 70)

    # H-TEMP-1: Classification waves
    print("\n  [H-TEMP-1] Classification Waves")
    waves_result = test_classification_waves(df)
    print(f"    Distribution: {waves_result.get('distribution', 'N/A')}")
    print(f"    p-value: {waves_result.get('ks_uniform_p', 'N/A')}")
    print(f"    Wave decades: {waves_result.get('wave_decades', [])}")
    print(f"    → {waves_result.get('h_temp_1', 'N/A')}")

    # H-TEMP-2: Mass trend
    print("\n  [H-TEMP-2] Mass Trend Over Time")
    mass_result = test_mass_trend(df)
    print(f"    Spearman r: {mass_result.get('spearman_r', 'N/A'):.3f}")
    print(f"    p-value: {mass_result.get('spearman_p', 'N/A'):.4f}")
    print(f"    Trend: {mass_result.get('trend', 'N/A')}")
    print(f"    → {mass_result.get('h_temp_2', 'N/A')}")

    # H-TEMP-3: Antarctica effect
    print("\n  [H-TEMP-3] Antarctica Effect")
    antarctica_result = test_antarctica_effect(df)
    print(f"    Pre-1970 median mass: {antarctica_result.get('pre_1970_median_mass', 0):,.0f}g")
    print(f"    Post-1970 median mass: {antarctica_result.get('post_1970_median_mass', 0):,.0f}g")
    print(f"    KS test p-value: {antarctica_result.get('mass_ks_p', 'N/A'):.2e}")
    print(f"    → {antarctica_result.get('h_temp_3', 'N/A')}")

    # Diversity growth
    print("\n  [DIVERSITY] Growth Pattern")
    diversity_result = test_diversity_growth(df)
    print(f"    Pattern: {diversity_result.get('pattern', 'N/A')}")
    print(f"    R² (linear): {diversity_result.get('r_squared_linear', 0):.3f}")
    print(f"    Interpretation: {diversity_result.get('interpretation', 'N/A')}")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 3: Change Point Detection
    # ════════════════════════════════════════════════════════════════════
    print()
    print("-" * 70)
    print("PHASE 3: Change Point Detection")
    print("-" * 70)

    mass_changepoints = detect_changepoints_mass(df)
    diversity_changepoints = detect_changepoints_diversity(df)

    print(f"\n  Mass change points: {len(mass_changepoints)}")
    for cp in mass_changepoints[:5]:
        print(f"    {cp['decade']}: {cp['type']} (magnitude: {cp['magnitude']:.2f})")

    print(f"\n  Diversity change points: {len(diversity_changepoints)}")
    for cp in diversity_changepoints[:5]:
        print(f"    {cp['decade']}: {cp['type']} (jump: {cp['jump']:.2f})")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "n_total": len(df),
            "year_range": [int(df["year"].min()), int(df["year"].max())],
        },
        "era_metrics": {k: v.to_dict() for k, v in era_metrics.items()},
        "hypothesis_tests": {
            "h_temp_1_waves": waves_result,
            "h_temp_2_mass_trend": mass_result,
            "h_temp_3_antarctica": antarctica_result,
            "diversity_growth": diversity_result,
        },
        "changepoints": {
            "mass": mass_changepoints,
            "diversity": diversity_changepoints,
        },
    }


def generate_outputs(results: Dict[str, Any], output_dir: Path = REPORTS) -> Dict[str, str]:
    """Generate O-Δ9 outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    # 1. JSON results
    json_path = output_dir / "O-D9_temporal_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    files["json"] = str(json_path)

    # 2. Era comparison CSV
    era_data = []
    for era_name, metrics in results["era_metrics"].items():
        era_data.append({
            "era": era_name,
            "year_start": metrics["year_range"][0],
            "year_end": metrics["year_range"][1],
            "n_total": metrics["n_total"],
            "n_falls": metrics["n_falls"],
            "n_finds": metrics["n_finds"],
            "fall_ratio": metrics["fall_ratio"],
            "n_classes": metrics["n_classes"],
            "n_new_classes": metrics["n_new_classes"],
            "shannon_diversity": metrics["shannon_diversity"],
            "mass_median": metrics["mass_median"],
            "mass_log_mean": metrics["mass_log_mean"],
        })

    era_df = pd.DataFrame(era_data)
    era_path = output_dir / "O-D9_era_comparison.csv"
    era_df.to_csv(era_path, index=False)
    files["era_csv"] = str(era_path)

    # 3. Hypothesis summary CSV
    hyp_data = [
        {
            "hypothesis": "H-TEMP-1",
            "description": "Classes appear in waves",
            "result": results["hypothesis_tests"]["h_temp_1_waves"].get("h_temp_1", "N/A"),
            "p_value": results["hypothesis_tests"]["h_temp_1_waves"].get("ks_uniform_p", np.nan),
        },
        {
            "hypothesis": "H-TEMP-2",
            "description": "Mass decreases over time",
            "result": results["hypothesis_tests"]["h_temp_2_mass_trend"].get("h_temp_2", "N/A"),
            "p_value": results["hypothesis_tests"]["h_temp_2_mass_trend"].get("spearman_p", np.nan),
        },
        {
            "hypothesis": "H-TEMP-3",
            "description": "Antarctica distorts catalog",
            "result": results["hypothesis_tests"]["h_temp_3_antarctica"].get("h_temp_3", "N/A"),
            "p_value": results["hypothesis_tests"]["h_temp_3_antarctica"].get("mass_ks_p", np.nan),
        },
    ]

    hyp_df = pd.DataFrame(hyp_data)
    hyp_path = output_dir / "O-D9_hypothesis_summary.csv"
    hyp_df.to_csv(hyp_path, index=False)
    files["hypothesis_csv"] = str(hyp_path)

    # 4. Generate plots
    plot_path = output_dir / "O-D9_temporal_analysis.png"
    generate_temporal_plots(results, plot_path)
    files["plot"] = str(plot_path)

    print()
    print("=" * 70)
    print("FILES GENERATED")
    print("=" * 70)
    for k, v in files.items():
        print(f"  {k}: {v}")

    return files


def generate_temporal_plots(results: Dict[str, Any], output_path: Path):
    """Generate temporal analysis plots."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ─────────────────────────────────────────────────────────────────
    # Plot 1: Era comparison
    # ─────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    eras = list(results["era_metrics"].keys())
    n_samples = [results["era_metrics"][e]["n_total"] for e in eras]
    colors = [ERA_COLORS[e] for e in eras]

    bars = ax1.bar(range(len(eras)), n_samples, color=colors)
    ax1.set_xticks(range(len(eras)))
    ax1.set_xticklabels([e.replace("_", "\n") for e in eras], fontsize=9)
    ax1.set_ylabel("Number of Samples")
    ax1.set_title("Samples by Era")

    for bar, n in zip(bars, n_samples):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{n:,}', ha='center', va='bottom', fontsize=9)

    # ─────────────────────────────────────────────────────────────────
    # Plot 2: Mass trend over time
    # ─────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    mass_result = results["hypothesis_tests"]["h_temp_2_mass_trend"]
    if "decades" in mass_result and "log_medians" in mass_result:
        decades = mass_result["decades"]
        log_medians = mass_result["log_medians"]
        medians = np.exp(log_medians)

        ax2.semilogy(decades, medians, 'o-', color='steelblue', linewidth=2)
        ax2.axvline(x=1970, color='red', linestyle='--', alpha=0.5, label='Antarctica (1970)')
        ax2.set_xlabel("Decade")
        ax2.set_ylabel("Median Mass (g, log scale)")
        ax2.set_title(f"Mass Trend: {mass_result.get('trend', 'N/A')}\n"
                     f"(Spearman r={mass_result.get('spearman_r', 0):.3f}, "
                     f"p={mass_result.get('spearman_p', 1):.4f})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # ─────────────────────────────────────────────────────────────────
    # Plot 3: Cumulative class discovery
    # ─────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    div_result = results["hypothesis_tests"]["diversity_growth"]
    if "decades" in div_result and "cumulative_classes" in div_result:
        decades = div_result["decades"]
        cum_classes = div_result["cumulative_classes"]

        ax3.plot(decades, cum_classes, 'o-', color='forestgreen', linewidth=2)
        ax3.axvline(x=1970, color='red', linestyle='--', alpha=0.5, label='Antarctica (1970)')
        ax3.set_xlabel("Decade")
        ax3.set_ylabel("Cumulative Number of Classes")
        ax3.set_title(f"Class Discovery: {div_result.get('pattern', 'N/A')}\n"
                     f"(R² linear={div_result.get('r_squared_linear', 0):.3f})")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # ─────────────────────────────────────────────────────────────────
    # Plot 4: New classes per decade
    # ─────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    waves_result = results["hypothesis_tests"]["h_temp_1_waves"]
    if "new_classes_by_decade" in waves_result:
        decade_counts = waves_result["new_classes_by_decade"]
        decades = sorted([int(k) for k in decade_counts.keys()])
        # Handle both string and int keys
        counts = [decade_counts.get(d, decade_counts.get(str(d), 0)) for d in decades]

        wave_decades = waves_result.get("wave_decades", [])
        colors = ['red' if d in wave_decades else 'steelblue' for d in decades]

        ax4.bar(decades, counts, color=colors, width=8)
        ax4.axvline(x=1970, color='green', linestyle='--', alpha=0.5, label='Antarctica (1970)')
        ax4.set_xlabel("Decade")
        ax4.set_ylabel("New Classes Discovered")
        ax4.set_title(f"Classification Waves: {waves_result.get('distribution', 'N/A')}\n"
                     f"(p={waves_result.get('ks_uniform_p', 1):.4f}, red=wave decades)")
        ax4.legend()

    plt.suptitle("O-Δ9: Catalog Archaeology — Temporal Dynamics",
                fontsize=14, fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_observation_md(results: Dict[str, Any], files: Dict[str, str], output_dir: Path) -> str:
    """Generate observation markdown."""
    date = datetime.now().strftime("%Y%m%d")
    md_path = output_dir / f"observation_O-D9_{date}.md"

    h1 = results["hypothesis_tests"]["h_temp_1_waves"]
    h2 = results["hypothesis_tests"]["h_temp_2_mass_trend"]
    h3 = results["hypothesis_tests"]["h_temp_3_antarctica"]
    div = results["hypothesis_tests"]["diversity_growth"]

    lines = [
        "# Observation O-Δ9: Catalog Archaeology",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Experiment**: O-Δ9 (Temporal Dynamics)",
        "",
        "---",
        "",
        "## Question",
        "",
        "Does the meteorite catalog reflect the universe,",
        "or does it reflect the history of human observation?",
        "",
        "---",
        "",
        "## Era Comparison",
        "",
        "| Era | Years | Samples | Classes | New Classes | Median Mass |",
        "|-----|-------|---------|---------|-------------|-------------|",
    ]

    for era_name, metrics in results["era_metrics"].items():
        yr = f"{metrics['year_range'][0]}-{metrics['year_range'][1]}"
        lines.append(
            f"| {era_name} | {yr} | {metrics['n_total']:,} | "
            f"{metrics['n_classes']} | {metrics['n_new_classes']} | "
            f"{metrics['mass_median']:,.0f}g |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Hypothesis Results",
        "",
        "| Hypothesis | Description | Result | p-value |",
        "|------------|-------------|--------|---------|",
        f"| H-TEMP-1 | Classes appear in waves | {h1.get('h_temp_1', 'N/A')} | {h1.get('ks_uniform_p', 'N/A'):.4f} |",
        f"| H-TEMP-2 | Mass decreases over time | {h2.get('h_temp_2', 'N/A')} | {h2.get('spearman_p', 'N/A'):.4f} |",
        f"| H-TEMP-3 | Antarctica distorts catalog | {h3.get('h_temp_3', 'N/A')} | {h3.get('mass_ks_p', 'N/A'):.2e} |",
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ])

    # Key findings based on results
    if h2.get("h_temp_2") == "SUPPORTED":
        lines.append("1. **Mass DECREASES over time** — We found the big ones first")

    if h3.get("h_temp_3") == "SUPPORTED":
        lines.append("2. **Antarctica dramatically changed the catalog** — Post-1970 distribution is fundamentally different")

    if h1.get("h_temp_1") == "SUPPORTED":
        lines.append(f"3. **Classification happens in waves** — Wave decades: {h1.get('wave_decades', [])}")

    if div.get("pattern") == "LINEAR_GROWTH":
        lines.append("4. **Class discovery continues** — We haven't found them all yet")
    elif div.get("pattern") == "SATURATING":
        lines.append("4. **Class discovery saturating** — Approaching true diversity")

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
        "*Generated by ORIGINMAP O-Δ9 experiment*",
    ])

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return str(md_path)


def run_full_o_delta_9() -> Dict[str, str]:
    """Main entry point for O-Δ9."""
    results = run_o_delta_9()
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
    print("O-Δ9 VERDICT")
    print("=" * 70)

    h1 = results["hypothesis_tests"]["h_temp_1_waves"]
    h2 = results["hypothesis_tests"]["h_temp_2_mass_trend"]
    h3 = results["hypothesis_tests"]["h_temp_3_antarctica"]

    print(f"\n  H-TEMP-1 (waves):     {h1.get('h_temp_1', 'N/A')}")
    print(f"  H-TEMP-2 (mass↓):     {h2.get('h_temp_2', 'N/A')}")
    print(f"  H-TEMP-3 (antarctica): {h3.get('h_temp_3', 'N/A')}")

    supported = sum(1 for h in [h1.get('h_temp_1'), h2.get('h_temp_2'), h3.get('h_temp_3')]
                   if h == "SUPPORTED")

    print()
    if supported >= 2:
        print("  → CATALOG REFLECTS OBSERVATION HISTORY, NOT UNIVERSE")
    elif supported == 1:
        print("  → MIXED: Some observational bias detected")
    else:
        print("  → CATALOG MAY REFLECT UNIVERSE (no strong bias)")

    return files


if __name__ == "__main__":
    run_full_o_delta_9()
