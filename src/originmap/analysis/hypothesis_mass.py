"""
Hypothesis Engine - Mass Heterogeneity (O-Δ2)

Question: Is mass variance within each class different from what we'd
expect under random assignment?

This tests whether certain meteorite classes have "unusual" mass
distributions (too tight or too dispersed) compared to a null model.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from originmap.config import DATA_PROCESSED, REPORTS


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def coefficient_of_variation(values: np.ndarray) -> float:
    """Calculate CV = std / mean (scale-invariant dispersion)."""
    if len(values) < 2:
        return np.nan
    mean = np.mean(values)
    if mean == 0:
        return np.nan
    return np.std(values) / mean


def log_range(values: np.ndarray) -> float:
    """Calculate log10(max/min) — captures order-of-magnitude spread."""
    values = values[values > 0]
    if len(values) < 2:
        return np.nan
    return np.log10(values.max() / values.min())


def calculate_observed_mass_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate observed mass heterogeneity metrics for each class."""
    results = []

    for recclass, group in df.groupby("recclass"):
        masses = group["mass"].dropna().values
        if len(masses) < 5:  # Need minimum samples for meaningful stats
            continue

        results.append({
            "recclass": recclass,
            "count": len(masses),
            "mass_mean": np.mean(masses),
            "mass_median": np.median(masses),
            "mass_std": np.std(masses),
            "cv_obs": coefficient_of_variation(masses),
            "log_range_obs": log_range(masses),
            "iqr_obs": np.percentile(masses, 75) - np.percentile(masses, 25),
        })

    return pd.DataFrame(results)


def run_single_permutation(
    all_masses: np.ndarray,
    class_sizes: Dict[str, int],
    rng: np.random.Generator
) -> Dict[str, Dict[str, float]]:
    """
    Run one permutation: shuffle masses globally, then calculate
    stats per class using original class sizes.

    Null model preserves:
    - Class sizes
    - Global mass distribution

    Null model breaks:
    - Specific mass ↔ class association
    """
    shuffled_masses = all_masses.copy()
    rng.shuffle(shuffled_masses)

    results = {}
    idx = 0

    for recclass, size in class_sizes.items():
        class_masses = shuffled_masses[idx:idx + size]
        idx += size

        results[recclass] = {
            "cv": coefficient_of_variation(class_masses),
            "log_range": log_range(class_masses),
            "iqr": np.percentile(class_masses, 75) - np.percentile(class_masses, 25) if len(class_masses) >= 4 else np.nan,
        }

    return results


def run_permutation_test(
    df: pd.DataFrame,
    n_permutations: int = 1000,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run permutation test for mass heterogeneity.
    """
    rng = np.random.default_rng(seed)

    # Filter to valid data
    df = df.dropna(subset=["mass", "recclass"])
    df = df[df["mass"] > 0]

    # Calculate observed stats
    obs_df = calculate_observed_mass_stats(df)

    # Get class sizes (only for classes with enough samples)
    valid_classes = set(obs_df["recclass"])
    df_filtered = df[df["recclass"].isin(valid_classes)]

    class_sizes = obs_df.set_index("recclass")["count"].to_dict()
    all_masses = df_filtered["mass"].values

    # Initialize null distributions
    null_cv = {rc: [] for rc in class_sizes.keys()}
    null_log_range = {rc: [] for rc in class_sizes.keys()}
    null_iqr = {rc: [] for rc in class_sizes.keys()}

    # Run permutations
    print(f"  Running {n_permutations} permutations...")
    for i in range(n_permutations):
        if i % 200 == 0:
            print(f"    Permutation {i}/{n_permutations}")

        perm_results = run_single_permutation(all_masses, class_sizes, rng)

        for recclass, metrics in perm_results.items():
            null_cv[recclass].append(metrics["cv"])
            null_log_range[recclass].append(metrics["log_range"])
            null_iqr[recclass].append(metrics["iqr"])

    # Calculate statistics for each class
    results = []

    for _, row in obs_df.iterrows():
        rc = row["recclass"]

        # CV statistics
        cv_null = np.array(null_cv[rc])
        cv_null = cv_null[~np.isnan(cv_null)]
        cv_obs = row["cv_obs"]

        if len(cv_null) > 10 and not np.isnan(cv_obs):
            cv_mean = np.mean(cv_null)
            cv_std = np.std(cv_null)

            # Two-sided p-value: is observed unusually high OR low?
            cv_pvalue_high = (cv_null >= cv_obs).sum() / len(cv_null)
            cv_pvalue_low = (cv_null <= cv_obs).sum() / len(cv_null)
            cv_pvalue = 2 * min(cv_pvalue_high, cv_pvalue_low)  # Two-sided

            cv_zscore = (cv_obs - cv_mean) / cv_std if cv_std > 0 else 0
            cv_effect = cv_obs - cv_mean
        else:
            cv_mean = cv_std = cv_pvalue = cv_zscore = cv_effect = np.nan

        # Log-range statistics
        lr_null = np.array(null_log_range[rc])
        lr_null = lr_null[~np.isnan(lr_null)]
        lr_obs = row["log_range_obs"]

        if len(lr_null) > 10 and not np.isnan(lr_obs):
            lr_mean = np.mean(lr_null)
            lr_std = np.std(lr_null)

            lr_pvalue_high = (lr_null >= lr_obs).sum() / len(lr_null)
            lr_pvalue_low = (lr_null <= lr_obs).sum() / len(lr_null)
            lr_pvalue = 2 * min(lr_pvalue_high, lr_pvalue_low)

            lr_zscore = (lr_obs - lr_mean) / lr_std if lr_std > 0 else 0
            lr_effect = lr_obs - lr_mean
        else:
            lr_mean = lr_std = lr_pvalue = lr_zscore = lr_effect = np.nan

        results.append({
            "recclass": rc,
            "count": row["count"],
            "mass_mean": row["mass_mean"],
            "mass_median": row["mass_median"],
            # CV metrics
            "cv_obs": cv_obs,
            "cv_null_mean": cv_mean,
            "cv_null_std": cv_std,
            "cv_effect": cv_effect,
            "cv_zscore": cv_zscore,
            "cv_pvalue": cv_pvalue,
            # Log-range metrics
            "log_range_obs": lr_obs,
            "log_range_null_mean": lr_mean,
            "log_range_null_std": lr_std,
            "log_range_effect": lr_effect,
            "log_range_zscore": lr_zscore,
            "log_range_pvalue": lr_pvalue,
        })

    results_df = pd.DataFrame(results)

    # Apply FDR correction
    results_df = apply_fdr_correction(results_df, "cv_pvalue", "cv_fdr_q")
    results_df = apply_fdr_correction(results_df, "log_range_pvalue", "log_range_fdr_q")

    # Build summary
    summary = build_summary(results_df, n_permutations, seed)

    return results_df, summary


def apply_fdr_correction(df: pd.DataFrame, pvalue_col: str, q_col: str) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction."""
    df = df.copy()
    pvalues = df[pvalue_col].values

    # Handle NaNs
    valid_mask = ~np.isnan(pvalues)
    n_valid = valid_mask.sum()

    if n_valid == 0:
        df[q_col] = np.nan
        return df

    # Sort valid p-values
    valid_pvalues = pvalues[valid_mask]
    sorted_idx = np.argsort(valid_pvalues)
    sorted_pvalues = valid_pvalues[sorted_idx]

    # Calculate BH adjusted p-values
    q_values_valid = np.zeros(n_valid)
    cummin = 1.0

    for i in range(n_valid - 1, -1, -1):
        rank = i + 1
        bh_value = sorted_pvalues[i] * n_valid / rank
        cummin = min(cummin, bh_value)
        q_values_valid[sorted_idx[i]] = min(cummin, 1.0)

    # Map back to full array
    q_values = np.full(len(pvalues), np.nan)
    q_values[valid_mask] = q_values_valid
    df[q_col] = q_values

    return df


def build_summary(results_df: pd.DataFrame, n_permutations: int, seed: int) -> Dict[str, Any]:
    """Build summary statistics."""
    valid_cv = results_df.dropna(subset=["cv_fdr_q"])
    valid_lr = results_df.dropna(subset=["log_range_fdr_q"])

    # Discoveries
    cv_discoveries_005 = (valid_cv["cv_fdr_q"] <= 0.05).sum()
    cv_discoveries_010 = (valid_cv["cv_fdr_q"] <= 0.10).sum()
    lr_discoveries_005 = (valid_lr["log_range_fdr_q"] <= 0.05).sum()
    lr_discoveries_010 = (valid_lr["log_range_fdr_q"] <= 0.10).sum()

    # Top anomalous (low CV = too consistent, high CV = too dispersed)
    # Classes with unusually LOW variance (tight mass range)
    tight_classes = valid_cv[
        (valid_cv["cv_fdr_q"] <= 0.10) &
        (valid_cv["cv_effect"] < 0)
    ].nsmallest(10, "cv_pvalue")

    # Classes with unusually HIGH variance (dispersed mass range)
    dispersed_classes = valid_cv[
        (valid_cv["cv_fdr_q"] <= 0.10) &
        (valid_cv["cv_effect"] > 0)
    ].nsmallest(10, "cv_pvalue")

    return {
        "experiment": "O-Δ2",
        "hypothesis": "Mass heterogeneity matches random assignment",
        "null_model": "Masses randomly assigned to classes, preserving class sizes",
        "metric": "Coefficient of Variation (CV = std/mean)",
        "timestamp_utc": datetime.utcnow().isoformat(),
        "parameters": {
            "n_permutations": n_permutations,
            "seed": seed,
            "n_classes_tested": len(valid_cv),
        },
        "discoveries": {
            "cv_fdr_005": int(cv_discoveries_005),
            "cv_fdr_010": int(cv_discoveries_010),
            "log_range_fdr_005": int(lr_discoveries_005),
            "log_range_fdr_010": int(lr_discoveries_010),
        },
        "unusually_tight_mass_classes": tight_classes[
            ["recclass", "count", "cv_obs", "cv_null_mean", "cv_zscore", "cv_fdr_q"]
        ].to_dict(orient="records") if len(tight_classes) > 0 else [],
        "unusually_dispersed_mass_classes": dispersed_classes[
            ["recclass", "count", "cv_obs", "cv_null_mean", "cv_zscore", "cv_fdr_q"]
        ].to_dict(orient="records") if len(dispersed_classes) > 0 else [],
    }


def generate_plot(results_df: pd.DataFrame, output_path: Path) -> str:
    """Generate visualization of mass heterogeneity results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    valid = results_df.dropna(subset=["cv_fdr_q"])

    # Left: CV observed vs null
    ax1 = axes[0]
    significant = valid["cv_fdr_q"] <= 0.10

    ax1.scatter(
        valid.loc[~significant, "cv_null_mean"],
        valid.loc[~significant, "cv_obs"],
        alpha=0.3, s=20, c="gray", label="Not significant"
    )

    # Significant: color by direction
    sig_tight = significant & (valid["cv_effect"] < 0)
    sig_disp = significant & (valid["cv_effect"] > 0)

    ax1.scatter(
        valid.loc[sig_tight, "cv_null_mean"],
        valid.loc[sig_tight, "cv_obs"],
        alpha=0.8, s=50, c="blue", marker="v", label="Unusually tight"
    )
    ax1.scatter(
        valid.loc[sig_disp, "cv_null_mean"],
        valid.loc[sig_disp, "cv_obs"],
        alpha=0.8, s=50, c="red", marker="^", label="Unusually dispersed"
    )

    lims = [0, max(valid["cv_obs"].max(), valid["cv_null_mean"].max()) * 1.1]
    ax1.plot(lims, lims, 'k--', alpha=0.5)
    ax1.set_xlabel("CV (null mean)")
    ax1.set_ylabel("CV (observed)")
    ax1.set_title("O-Δ2: Mass Coefficient of Variation")
    ax1.legend(loc="upper left")

    # Right: Z-score distribution
    ax2 = axes[1]
    zscores = valid["cv_zscore"].dropna()

    ax2.hist(zscores, bins=30, alpha=0.7, edgecolor="black")
    ax2.axvline(x=-2, color="blue", linestyle="--", label="z = -2 (tight)")
    ax2.axvline(x=2, color="red", linestyle="--", label="z = +2 (dispersed)")
    ax2.set_xlabel("Z-score")
    ax2.set_ylabel("Number of classes")
    ax2.set_title("Distribution of CV Z-scores")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def run_experiment_o_delta_2(n_permutations: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Run the O-Δ2 mass heterogeneity experiment."""
    df = pd.read_parquet(DATA_PROCESSED / "meteorites.parquet")

    print(f"Running O-Δ2: Mass heterogeneity test")
    print(f"  Permutations: {n_permutations}, Seed: {seed}")

    results_df, summary = run_permutation_test(df, n_permutations, seed)

    # Save results
    REPORTS.mkdir(parents=True, exist_ok=True)

    csv_path = REPORTS / "hypothesis_O-Δ2_results.csv"
    results_df.to_csv(csv_path, index=False)

    json_path = REPORTS / "hypothesis_O-Δ2_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    plot_path = REPORTS / "hypothesis_O-Δ2_plot.png"
    generate_plot(results_df, plot_path)

    print(f"\n  CV discoveries (FDR≤0.05): {summary['discoveries']['cv_fdr_005']}")
    print(f"  CV discoveries (FDR≤0.10): {summary['discoveries']['cv_fdr_010']}")

    return {
        "csv": str(csv_path),
        "summary": str(json_path),
        "plot": str(plot_path),
        "discoveries": summary["discoveries"],
        "tight_classes": summary["unusually_tight_mass_classes"][:5],
        "dispersed_classes": summary["unusually_dispersed_mass_classes"][:5],
    }


if __name__ == "__main__":
    result = run_experiment_o_delta_2(n_permutations=500, seed=42)
    print(f"\nResults saved to: {result['csv']}")
