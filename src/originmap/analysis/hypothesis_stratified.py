"""
Hypothesis Engine - Stratified Null Model (O-Δ3)

This is the CRITICAL test that separates artifact from structure.

Null-1 (O-Δ2): Shuffle all masses globally
Null-2 (O-Δ3): Shuffle masses ONLY within mass bins

If a class survives Null-2, the tight variance is NOT because:
- They sample from a restricted mass range
- Selection bias toward certain mass ranges

It IS because:
- Intrinsic constraint on mass distribution within the class
- Physical formation/fragmentation process
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any, List
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
        return super().default(obj)


def coefficient_of_variation(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    mean = np.mean(values)
    if mean == 0:
        return np.nan
    return np.std(values) / mean


def assign_mass_bins(masses: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Assign each mass to a quantile bin.
    Returns array of bin indices (0 to n_bins-1).
    """
    # Use quantiles for equal-count bins
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(masses, quantiles)

    # Handle edge case: ensure max value is included
    bin_edges[-1] = bin_edges[-1] + 1

    bins = np.digitize(masses, bin_edges) - 1
    bins = np.clip(bins, 0, n_bins - 1)

    return bins


def run_stratified_permutation(
    df: pd.DataFrame,
    class_indices: Dict[str, List[int]],
    mass_bins: np.ndarray,
    rng: np.random.Generator
) -> Dict[str, float]:
    """
    Run one STRATIFIED permutation.

    Key difference from Null-1:
    - Masses are only shuffled WITHIN their bin
    - Each class gets masses from the same bins as original
    - But the specific masses within each bin are randomized

    This preserves:
    - Class sizes
    - Mass bin distribution per class
    - Overall mass distribution

    This breaks:
    - Specific mass ↔ class association within bins
    """
    masses = df["mass"].values.copy()
    n_bins = mass_bins.max() + 1

    # Shuffle within each bin
    shuffled_masses = masses.copy()
    for bin_idx in range(n_bins):
        bin_mask = mass_bins == bin_idx
        bin_indices = np.where(bin_mask)[0]
        if len(bin_indices) > 1:
            bin_values = masses[bin_indices].copy()
            rng.shuffle(bin_values)
            shuffled_masses[bin_indices] = bin_values

    # Calculate CV for each class using shuffled masses
    results = {}
    for recclass, indices in class_indices.items():
        class_masses = shuffled_masses[indices]
        results[recclass] = coefficient_of_variation(class_masses)

    return results


def run_experiment_o_delta_3(
    n_permutations: int = 500,
    seed: int = 42,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Run the O-Δ3 stratified null model experiment.

    This is the HARD test. Classes that survive this have genuine structure.
    """
    print(f"Running O-Δ3: Stratified null model")
    print(f"  Permutations: {n_permutations}, Seed: {seed}, Bins: {n_bins}")

    rng = np.random.default_rng(seed)

    # Load and prepare data
    df = pd.read_parquet(DATA_PROCESSED / "meteorites.parquet")
    df = df.dropna(subset=["mass", "recclass"])
    df = df[df["mass"] > 0].reset_index(drop=True)

    # Assign mass bins
    mass_bins = assign_mass_bins(df["mass"].values, n_bins)
    df["mass_bin"] = mass_bins

    # Build class indices (for efficiency)
    class_indices = {}
    for recclass in df["recclass"].unique():
        indices = df[df["recclass"] == recclass].index.tolist()
        if len(indices) >= 5:  # Minimum samples
            class_indices[recclass] = indices

    print(f"  Classes tested: {len(class_indices)}")

    # Calculate observed CV
    observed_cv = {}
    observed_stats = {}
    for recclass, indices in class_indices.items():
        masses = df.loc[indices, "mass"].values
        cv = coefficient_of_variation(masses)
        observed_cv[recclass] = cv
        observed_stats[recclass] = {
            "count": len(indices),
            "mean_mass": np.mean(masses),
            "median_mass": np.median(masses),
            "cv_obs": cv,
            # Track bin distribution for this class
            "bin_counts": np.bincount(mass_bins[indices], minlength=n_bins).tolist()
        }

    # Run stratified permutations
    null_cv = {rc: [] for rc in class_indices.keys()}

    print(f"  Running {n_permutations} stratified permutations...")
    for i in range(n_permutations):
        if i % 100 == 0:
            print(f"    Permutation {i}/{n_permutations}")

        perm_results = run_stratified_permutation(df, class_indices, mass_bins, rng)

        for recclass, cv in perm_results.items():
            null_cv[recclass].append(cv)

    # Calculate statistics
    results = []
    for recclass, stats in observed_stats.items():
        cv_obs = stats["cv_obs"]
        cv_null = np.array(null_cv[recclass])
        cv_null = cv_null[~np.isnan(cv_null)]

        if len(cv_null) < 10 or np.isnan(cv_obs):
            continue

        cv_mean = np.mean(cv_null)
        cv_std = np.std(cv_null)

        # Two-sided p-value
        pvalue_low = (cv_null <= cv_obs).sum() / len(cv_null)
        pvalue_high = (cv_null >= cv_obs).sum() / len(cv_null)
        pvalue = 2 * min(pvalue_low, pvalue_high)

        zscore = (cv_obs - cv_mean) / cv_std if cv_std > 0 else 0
        effect = cv_obs - cv_mean

        results.append({
            "recclass": recclass,
            "count": stats["count"],
            "mean_mass": stats["mean_mass"],
            "cv_obs": cv_obs,
            "cv_null_mean": cv_mean,
            "cv_null_std": cv_std,
            "cv_effect": effect,
            "cv_zscore": zscore,
            "cv_pvalue": pvalue,
            "bin_counts": stats["bin_counts"],
        })

    results_df = pd.DataFrame(results)

    # FDR correction
    results_df = apply_fdr_correction(results_df)

    # Build summary
    summary = build_summary(results_df, n_permutations, seed, n_bins)

    # Save outputs
    REPORTS.mkdir(parents=True, exist_ok=True)

    csv_path = REPORTS / "hypothesis_O-Δ3_results.csv"
    # Don't save bin_counts in CSV (it's a list)
    results_df.drop(columns=["bin_counts"]).to_csv(csv_path, index=False)

    json_path = REPORTS / "hypothesis_O-Δ3_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    plot_path = REPORTS / "hypothesis_O-Δ3_plot.png"
    generate_comparison_plot(results_df, plot_path)

    print(f"\n  Discoveries (FDR≤0.05): {summary['discoveries']['fdr_005']}")
    print(f"  Discoveries (FDR≤0.10): {summary['discoveries']['fdr_010']}")

    return {
        "csv": str(csv_path),
        "summary": str(json_path),
        "plot": str(plot_path),
        "discoveries": summary["discoveries"],
        "survivors": summary["survivors_from_od2"],
        "fallen": summary["fallen_from_od2"],
    }


def apply_fdr_correction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pvalues = df["cv_pvalue"].values
    valid_mask = ~np.isnan(pvalues)
    n_valid = valid_mask.sum()

    if n_valid == 0:
        df["cv_fdr_q"] = np.nan
        return df

    valid_pvalues = pvalues[valid_mask]
    sorted_idx = np.argsort(valid_pvalues)
    sorted_pvalues = valid_pvalues[sorted_idx]

    q_values_valid = np.zeros(n_valid)
    cummin = 1.0

    for i in range(n_valid - 1, -1, -1):
        rank = i + 1
        bh_value = sorted_pvalues[i] * n_valid / rank
        cummin = min(cummin, bh_value)
        q_values_valid[sorted_idx[i]] = min(cummin, 1.0)

    q_values = np.full(len(pvalues), np.nan)
    q_values[valid_mask] = q_values_valid
    df["cv_fdr_q"] = q_values

    return df


def build_summary(
    results_df: pd.DataFrame,
    n_permutations: int,
    seed: int,
    n_bins: int
) -> Dict[str, Any]:
    """Build summary comparing O-Δ2 and O-Δ3 results."""

    valid = results_df.dropna(subset=["cv_fdr_q"])

    # Load O-Δ2 results for comparison
    od2_path = REPORTS / "hypothesis_O-Δ2_results.csv"
    od2_survivors = set()
    if od2_path.exists():
        od2_df = pd.read_csv(od2_path)
        od2_sig = od2_df[od2_df["cv_fdr_q"] <= 0.10]
        od2_survivors = set(od2_sig["recclass"])

    # Current O-Δ3 significant classes
    od3_sig = valid[(valid["cv_fdr_q"] <= 0.10) & (valid["cv_effect"] < 0)]
    od3_survivors = set(od3_sig["recclass"])

    # Who survived from O-Δ2 to O-Δ3?
    still_significant = od2_survivors & od3_survivors
    fallen = od2_survivors - od3_survivors

    discoveries_005 = (valid["cv_fdr_q"] <= 0.05).sum()
    discoveries_010 = (valid["cv_fdr_q"] <= 0.10).sum()

    # Get details of survivors
    survivor_details = od3_sig[od3_sig["recclass"].isin(still_significant)][
        ["recclass", "count", "cv_obs", "cv_null_mean", "cv_zscore", "cv_fdr_q"]
    ].sort_values("cv_zscore").to_dict(orient="records")

    # Get details of fallen
    fallen_details = []
    if len(fallen) > 0:
        fallen_rows = valid[valid["recclass"].isin(fallen)]
        fallen_details = fallen_rows[
            ["recclass", "count", "cv_obs", "cv_null_mean", "cv_zscore", "cv_fdr_q"]
        ].to_dict(orient="records")

    return {
        "experiment": "O-Δ3",
        "hypothesis": "H-CONS: Constrained Formation",
        "null_model": "Stratified by mass bins - shuffles only within quantile bins",
        "comparison": "Classes that survive this harder null have genuine structure",
        "timestamp_utc": datetime.utcnow().isoformat(),
        "parameters": {
            "n_permutations": n_permutations,
            "seed": seed,
            "n_bins": n_bins,
            "n_classes_tested": len(valid),
        },
        "discoveries": {
            "fdr_005": int(discoveries_005),
            "fdr_010": int(discoveries_010),
        },
        "comparison_with_od2": {
            "od2_significant": len(od2_survivors),
            "od3_significant": len(od3_survivors),
            "survived": len(still_significant),
            "fallen": len(fallen),
        },
        "survivors_from_od2": survivor_details,
        "fallen_from_od2": fallen_details,
    }


def generate_comparison_plot(results_df: pd.DataFrame, output_path: Path) -> str:
    """Generate visualization comparing O-Δ2 and O-Δ3."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    valid = results_df.dropna(subset=["cv_fdr_q"])

    # Load O-Δ2 for comparison
    od2_path = REPORTS / "hypothesis_O-Δ2_results.csv"
    od2_survivors = set()
    if od2_path.exists():
        od2_df = pd.read_csv(od2_path)
        od2_sig = od2_df[od2_df["cv_fdr_q"] <= 0.10]
        od2_survivors = set(od2_sig["recclass"])

    # Left plot: O-Δ3 results
    ax1 = axes[0]

    sig = valid["cv_fdr_q"] <= 0.10
    survived_from_od2 = valid["recclass"].isin(od2_survivors) & sig
    new_sig = sig & ~valid["recclass"].isin(od2_survivors)

    ax1.scatter(
        valid.loc[~sig, "cv_null_mean"],
        valid.loc[~sig, "cv_obs"],
        alpha=0.3, s=20, c="gray", label="Not significant"
    )
    ax1.scatter(
        valid.loc[survived_from_od2, "cv_null_mean"],
        valid.loc[survived_from_od2, "cv_obs"],
        alpha=0.9, s=60, c="green", marker="*", label="Survived from O-Δ2"
    )
    ax1.scatter(
        valid.loc[new_sig, "cv_null_mean"],
        valid.loc[new_sig, "cv_obs"],
        alpha=0.8, s=40, c="blue", label="New in O-Δ3"
    )

    lims = [0, max(valid["cv_obs"].max(), valid["cv_null_mean"].max()) * 1.1]
    ax1.plot(lims, lims, 'k--', alpha=0.5)
    ax1.set_xlabel("CV (stratified null mean)")
    ax1.set_ylabel("CV (observed)")
    ax1.set_title("O-Δ3: Stratified Null Model")
    ax1.legend(loc="upper left")

    # Right plot: Z-score comparison
    ax2 = axes[1]

    # Show distribution
    zscores = valid["cv_zscore"].dropna()
    ax2.hist(zscores, bins=30, alpha=0.7, edgecolor="black", color="steelblue")
    ax2.axvline(x=-2, color="green", linestyle="--", linewidth=2, label="z = -2 (tight)")
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax2.axvline(x=2, color="red", linestyle="--", linewidth=2, label="z = +2 (dispersed)")

    # Mark survivors
    survivors = valid[valid["recclass"].isin(od2_survivors) & (valid["cv_fdr_q"] <= 0.10)]
    for _, row in survivors.iterrows():
        ax2.axvline(x=row["cv_zscore"], color="green", alpha=0.5, linewidth=2)

    ax2.set_xlabel("Z-score (stratified null)")
    ax2.set_ylabel("Number of classes")
    ax2.set_title("Z-score Distribution (green = O-Δ2 survivors)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


if __name__ == "__main__":
    result = run_experiment_o_delta_3(n_permutations=500, seed=42, n_bins=10)
    print(f"\nResults: {result['csv']}")
    print(f"Survivors from O-Δ2: {len(result['survivors'])}")
    print(f"Fallen from O-Δ2: {len(result['fallen'])}")
