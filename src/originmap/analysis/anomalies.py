"""
Anomaly detection module for ORIGINMAP.
Detects: heavy tails, diversity scaling issues, temporal asymmetries, rare invariants.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from math import log
from typing import Dict, List, Any
from originmap.config import DATA_PROCESSED, REPORTS


def load_dataset() -> pd.DataFrame:
    """Load the processed meteorite dataset."""
    path = DATA_PROCESSED / "meteorites.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_parquet(path)


def detect_heavy_tails(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect heavy tails in mass distribution.
    Uses kurtosis, skewness, and power-law fit quality.
    """
    findings = {"type": "heavy_tails", "anomalies": []}

    if "mass" not in df.columns:
        return findings

    mass = df["mass"].dropna()
    if len(mass) < 100:
        return findings

    # Basic statistics
    skewness = stats.skew(mass)
    kurtosis = stats.kurtosis(mass)

    # Extreme value analysis
    q99 = mass.quantile(0.99)
    q999 = mass.quantile(0.999)
    max_val = mass.max()
    median_val = mass.median()
    mean_val = mass.mean()

    # Ratio of mean to median (high = heavy tail)
    mean_median_ratio = mean_val / median_val if median_val > 0 else 0

    # Check for power-law behavior in log-log space
    log_mass = np.log10(mass[mass > 0])

    findings["statistics"] = {
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "mean_median_ratio": float(mean_median_ratio),
        "q99": float(q99),
        "q999": float(q999),
        "max": float(max_val),
        "median": float(median_val),
    }

    # Anomaly: extremely heavy tail
    if mean_median_ratio > 100:
        findings["anomalies"].append({
            "description": "Extreme mean/median ratio",
            "value": mean_median_ratio,
            "expected": "< 10 for normal distributions",
            "significance": "high"
        })

    # Anomaly: super-heavy kurtosis
    if kurtosis > 100:
        findings["anomalies"].append({
            "description": "Extreme kurtosis (leptokurtic)",
            "value": kurtosis,
            "expected": "~3 for normal, < 50 typical",
            "significance": "high"
        })

    # Check classes dominating extremes
    if "recclass" in df.columns:
        top_1pct = df[df["mass"] >= q99]
        bottom_50pct = df[df["mass"] <= median_val]

        top_classes = top_1pct["recclass"].value_counts(normalize=True).head(5)
        bottom_classes = bottom_50pct["recclass"].value_counts(normalize=True).head(5)

        # Check if same classes dominate both extremes
        top_set = set(top_classes.index)
        bottom_set = set(bottom_classes.index)

        if top_set != bottom_set:
            findings["class_asymmetry"] = {
                "top_1pct_dominant": top_classes.to_dict(),
                "bottom_50pct_dominant": bottom_classes.to_dict(),
                "overlap": list(top_set & bottom_set)
            }

            if len(top_set & bottom_set) == 0:
                findings["anomalies"].append({
                    "description": "Complete class separation between mass extremes",
                    "value": "no overlap in top-5 classes",
                    "expected": "some overlap",
                    "significance": "medium"
                })

    return findings


def detect_diversity_scaling(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect anomalies in diversity vs sample size scaling.
    Small classes with high diversity are non-trivial.
    """
    findings = {"type": "diversity_scaling", "anomalies": []}

    if "recclass" not in df.columns:
        return findings

    # Load diversity data if exists
    diversity_path = REPORTS / "diversity_by_class.csv"
    if diversity_path.exists():
        div_df = pd.read_csv(diversity_path)
    else:
        # Calculate on the fly
        class_counts = df["recclass"].value_counts()
        div_df = pd.DataFrame({
            "recclass": class_counts.index,
            "count": class_counts.values
        })

    # Group classes by size
    if "count" in div_df.columns:
        small_classes = div_df[div_df["count"] <= 10]
        medium_classes = div_df[(div_df["count"] > 10) & (div_df["count"] <= 100)]
        large_classes = div_df[div_df["count"] > 100]

        findings["size_distribution"] = {
            "small_n<=10": len(small_classes),
            "medium_10<n<=100": len(medium_classes),
            "large_n>100": len(large_classes),
            "total_classes": len(div_df)
        }

        # Anomaly: too many singleton classes
        singletons = div_df[div_df["count"] == 1]
        singleton_ratio = len(singletons) / len(div_df) if len(div_df) > 0 else 0

        if singleton_ratio > 0.3:
            findings["anomalies"].append({
                "description": "High singleton class ratio",
                "value": f"{singleton_ratio:.2%}",
                "expected": "< 20%",
                "significance": "medium",
                "classes": list(singletons["recclass"].head(10))
            })

    # Check for classes that appear once but are structurally unique
    if "mass" in df.columns:
        class_mass_stats = df.groupby("recclass")["mass"].agg(["count", "mean", "std"])

        # Classes with exactly 1 sample but extreme mass
        single_extreme = class_mass_stats[
            (class_mass_stats["count"] == 1) &
            (class_mass_stats["mean"] > df["mass"].quantile(0.99))
        ]

        if len(single_extreme) > 0:
            findings["anomalies"].append({
                "description": "Singleton classes with extreme mass",
                "value": len(single_extreme),
                "classes": list(single_extreme.index[:5]),
                "significance": "low"
            })

    return findings


def detect_temporal_asymmetries(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect temporal patterns: epochs, abrupt appearances/disappearances.
    """
    findings = {"type": "temporal_asymmetries", "anomalies": []}

    if "year" not in df.columns:
        return findings

    years = df["year"].dropna()
    if len(years) < 100:
        return findings

    # Define epochs
    epochs = {
        "pre_1900": (years < 1900).sum(),
        "1900_1950": ((years >= 1900) & (years < 1950)).sum(),
        "1950_1970": ((years >= 1950) & (years < 1970)).sum(),
        "1970_1990": ((years >= 1970) & (years < 1990)).sum(),
        "1990_2010": ((years >= 1990) & (years < 2010)).sum(),
        "post_2010": (years >= 2010).sum(),
    }

    findings["epoch_distribution"] = epochs

    # Calculate growth rates between epochs
    epoch_values = list(epochs.values())
    growth_rates = []
    for i in range(1, len(epoch_values)):
        if epoch_values[i-1] > 0:
            rate = epoch_values[i] / epoch_values[i-1]
            growth_rates.append(rate)

    findings["growth_rates"] = growth_rates

    # Anomaly: non-monotonic growth (decrease then increase)
    if len(growth_rates) >= 3:
        has_decrease = any(r < 0.8 for r in growth_rates)
        has_increase_after = False
        for i, r in enumerate(growth_rates):
            if r < 0.8 and i < len(growth_rates) - 1:
                if any(growth_rates[j] > 1.5 for j in range(i+1, len(growth_rates))):
                    has_increase_after = True

        if has_decrease and has_increase_after:
            findings["anomalies"].append({
                "description": "Non-monotonic temporal pattern",
                "value": "decrease followed by increase",
                "expected": "monotonic growth due to better detection",
                "significance": "medium"
            })

    # Check class appearances by epoch
    if "recclass" in df.columns:
        df_with_year = df.dropna(subset=["year", "recclass"])

        # First appearance of each class
        first_appearance = df_with_year.groupby("recclass")["year"].min()

        # Classes that first appear after 1990 (potentially new discoveries)
        new_classes = first_appearance[first_appearance >= 1990]
        old_classes = first_appearance[first_appearance < 1950]

        findings["class_temporality"] = {
            "first_appeared_post_1990": len(new_classes),
            "first_appeared_pre_1950": len(old_classes),
            "newest_classes": new_classes.sort_values(ascending=False).head(5).to_dict()
        }

        # Anomaly: classes that appear only in narrow time windows
        class_time_span = df_with_year.groupby("recclass")["year"].agg(["min", "max", "count"])
        class_time_span["span"] = class_time_span["max"] - class_time_span["min"]

        # Classes with multiple samples but narrow time span
        narrow_span = class_time_span[
            (class_time_span["count"] >= 5) &
            (class_time_span["span"] <= 10)
        ]

        if len(narrow_span) > 0:
            findings["anomalies"].append({
                "description": "Classes concentrated in narrow time windows",
                "value": len(narrow_span),
                "classes": list(narrow_span.index[:5]),
                "significance": "medium"
            })

    return findings


def detect_geographic_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect geographic clustering anomalies.
    """
    findings = {"type": "geographic_patterns", "anomalies": []}

    if "reclat" not in df.columns or "reclong" not in df.columns:
        return findings

    coords = df[["reclat", "reclong", "recclass"]].dropna()
    if len(coords) < 100:
        return findings

    # Hemisphere distribution
    northern = (coords["reclat"] > 0).sum()
    southern = (coords["reclat"] <= 0).sum()

    findings["hemisphere_distribution"] = {
        "northern": int(northern),
        "southern": int(southern),
        "ratio": float(northern / southern) if southern > 0 else float("inf")
    }

    # Anomaly: extreme hemisphere bias
    ratio = northern / southern if southern > 0 else 999
    if ratio > 3 or ratio < 0.33:
        findings["anomalies"].append({
            "description": "Extreme hemisphere bias in finds",
            "value": f"N/S ratio: {ratio:.2f}",
            "expected": "~1.0 for uniform distribution",
            "significance": "low",  # Likely observational bias
            "note": "Probably reflects Antarctic collection programs"
        })

    # Check for latitude bands with unusual class distributions
    coords["lat_band"] = pd.cut(coords["reclat"], bins=[-90, -60, -30, 0, 30, 60, 90])
    band_class_diversity = coords.groupby("lat_band", observed=False)["recclass"].nunique()

    # Convert Interval keys to strings for JSON serialization
    findings["diversity_by_latitude"] = {
        str(k): v for k, v in band_class_diversity.to_dict().items()
    }

    return findings


def detect_rare_invariants(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect unexpected invariants or correlations.
    """
    findings = {"type": "rare_invariants", "anomalies": []}

    # Check mass-class relationships
    if "mass" in df.columns and "recclass" in df.columns:
        class_mass = df.groupby("recclass")["mass"].agg(["mean", "std", "count"])
        class_mass = class_mass[class_mass["count"] >= 10]  # Only stable estimates

        # Coefficient of variation by class
        class_mass["cv"] = class_mass["std"] / class_mass["mean"]

        # Classes with unusually low CV (consistent mass)
        low_cv = class_mass[class_mass["cv"] < 0.5].sort_values("cv")

        if len(low_cv) > 0:
            findings["low_variance_classes"] = {
                "count": len(low_cv),
                "classes": low_cv.head(5).to_dict()
            }

            findings["anomalies"].append({
                "description": "Classes with unusually consistent mass",
                "value": len(low_cv),
                "classes": list(low_cv.index[:5]),
                "significance": "low"
            })

        # Classes with extremely high CV
        high_cv = class_mass[class_mass["cv"] > 5].sort_values("cv", ascending=False)

        if len(high_cv) > 0:
            findings["high_variance_classes"] = {
                "count": len(high_cv),
                "classes": high_cv.head(5).to_dict()
            }

    # Check year-mass correlation (should be ~0 if no bias)
    if "year" in df.columns and "mass" in df.columns:
        valid = df[["year", "mass"]].dropna()
        # Need variance in both columns for correlation
        if len(valid) > 100 and valid["year"].std() > 0 and valid["mass"].std() > 0:
            try:
                corr, pval = stats.pearsonr(valid["year"], valid["mass"])

                if not np.isnan(corr):
                    findings["year_mass_correlation"] = {
                        "pearson_r": float(corr),
                        "p_value": float(pval)
                    }

                    if abs(corr) > 0.1 and pval < 0.01:
                        findings["anomalies"].append({
                            "description": "Significant year-mass correlation",
                            "value": f"r={corr:.3f}, p={pval:.2e}",
                            "expected": "r~0 (no temporal bias)",
                            "significance": "medium"
                        })
            except Exception:
                pass  # Skip if correlation fails

    return findings


def run_all_detections() -> Dict[str, Any]:
    """Run all anomaly detection modules."""
    df = load_dataset()

    results = {
        "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
        "detections": {}
    }

    detectors = [
        detect_heavy_tails,
        detect_diversity_scaling,
        detect_temporal_asymmetries,
        detect_geographic_patterns,
        detect_rare_invariants,
    ]

    for detector in detectors:
        try:
            result = detector(df)
            results["detections"][result["type"]] = result
        except Exception as e:
            results["detections"][detector.__name__] = {"error": str(e)}

    # Collect all anomalies
    all_anomalies = []
    for det_name, det_result in results["detections"].items():
        if "anomalies" in det_result:
            for anom in det_result["anomalies"]:
                anom["detector"] = det_name
                all_anomalies.append(anom)

    results["all_anomalies"] = all_anomalies
    results["anomaly_count"] = len(all_anomalies)

    return results
