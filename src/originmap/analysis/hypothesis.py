"""
Hypothesis Engine for ORIGINMAP.
Rigorous permutation-based testing without ML.

Experiment O-Δ1: Does diversity scale proportionally with sample size?
Null model: Names randomly assigned to classes, preserving class sizes.
"""
import json
import numpy as np
import pandas as pd
from math import log
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
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


def shannon_entropy(names: pd.Series) -> float:
    """Calculate Shannon entropy for a series of names."""
    if len(names) == 0:
        return 0.0
    counts = names.value_counts()
    probs = counts / counts.sum()
    return -sum(p * log(p) for p in probs if p > 0)


def simpson_index(names: pd.Series) -> float:
    """Calculate Simpson diversity index (1 - D)."""
    if len(names) == 0:
        return 0.0
    counts = names.value_counts()
    probs = counts / counts.sum()
    return 1 - sum(p**2 for p in probs)


def calculate_observed_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate observed diversity metrics for each recclass.

    Returns DataFrame with columns:
    - recclass, count, unique_names, shannon_obs, simpson_obs
    """
    results = []

    for recclass, group in df.groupby("recclass"):
        names = group["name"]
        results.append({
            "recclass": recclass,
            "count": len(group),
            "unique_names": names.nunique(),
            "shannon_obs": shannon_entropy(names),
            "simpson_obs": simpson_index(names),
        })

    return pd.DataFrame(results)


def run_single_permutation(
    df: pd.DataFrame,
    class_sizes: Dict[str, int],
    rng: np.random.Generator
) -> Dict[str, Tuple[float, float]]:
    """
    Run one permutation: shuffle names across all samples, then calculate
    diversity per class using the original class sizes.

    The null model:
    - Preserves: class sizes (count per recclass), total pool of names
    - Breaks: specific name ↔ recclass association

    Returns dict: {recclass: (shannon, simpson)}
    """
    # Get all names and shuffle them
    all_names = df["name"].values.copy()
    rng.shuffle(all_names)

    # Assign shuffled names back to classes according to original sizes
    results = {}
    idx = 0

    for recclass, size in class_sizes.items():
        permuted_names = pd.Series(all_names[idx:idx + size])
        idx += size
        results[recclass] = (
            shannon_entropy(permuted_names),
            simpson_index(permuted_names)
        )

    return results


def run_permutation_test(
    df: pd.DataFrame,
    n_permutations: int = 1000,
    seed: int = 42,
    progress_callback=None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run full permutation test for experiment O-Δ1.

    Args:
        df: DataFrame with 'name' and 'recclass' columns
        n_permutations: Number of permutations (recommended: 500-2000)
        seed: Random seed for reproducibility
        progress_callback: Optional function(i, n) for progress updates

    Returns:
        results_df: Per-class results with p-values, z-scores, effect sizes
        summary: Global summary statistics
    """
    rng = np.random.default_rng(seed)

    # Calculate observed diversity
    obs_df = calculate_observed_diversity(df)

    # Get class sizes (this is what the null model preserves)
    class_sizes = dict(zip(obs_df["recclass"], obs_df["count"]))

    # Initialize null distribution storage
    null_shannon = {rc: [] for rc in class_sizes.keys()}
    null_simpson = {rc: [] for rc in class_sizes.keys()}

    # Run permutations
    for i in range(n_permutations):
        if progress_callback and i % 100 == 0:
            progress_callback(i, n_permutations)

        perm_results = run_single_permutation(df, class_sizes, rng)

        for recclass, (sh, si) in perm_results.items():
            null_shannon[recclass].append(sh)
            null_simpson[recclass].append(si)

    # Calculate statistics for each class
    results = []

    for _, row in obs_df.iterrows():
        rc = row["recclass"]

        # Shannon statistics
        sh_null = np.array(null_shannon[rc])
        sh_obs = row["shannon_obs"]
        sh_mean = sh_null.mean()
        sh_std = sh_null.std()

        # One-sided p-value: P(null >= obs) — tests if observed is unusually HIGH
        sh_pvalue = (sh_null >= sh_obs).sum() / n_permutations
        sh_zscore = (sh_obs - sh_mean) / sh_std if sh_std > 0 else 0
        sh_effect = sh_obs - sh_mean

        # Simpson statistics
        si_null = np.array(null_simpson[rc])
        si_obs = row["simpson_obs"]
        si_mean = si_null.mean()
        si_std = si_null.std()

        si_pvalue = (si_null >= si_obs).sum() / n_permutations
        si_zscore = (si_obs - si_mean) / si_std if si_std > 0 else 0
        si_effect = si_obs - si_mean

        results.append({
            "recclass": rc,
            "count": row["count"],
            "unique_names": row["unique_names"],
            # Shannon
            "shannon_obs": sh_obs,
            "shannon_null_mean": sh_mean,
            "shannon_null_std": sh_std,
            "shannon_effect": sh_effect,
            "shannon_zscore": sh_zscore,
            "shannon_pvalue": sh_pvalue,
            # Simpson
            "simpson_obs": si_obs,
            "simpson_null_mean": si_mean,
            "simpson_null_std": si_std,
            "simpson_effect": si_effect,
            "simpson_zscore": si_zscore,
            "simpson_pvalue": si_pvalue,
        })

    results_df = pd.DataFrame(results)

    # Apply FDR correction (Benjamini-Hochberg)
    results_df = apply_fdr_correction(results_df, "shannon_pvalue", "shannon_fdr_q")
    results_df = apply_fdr_correction(results_df, "simpson_pvalue", "simpson_fdr_q")

    # Build summary
    summary = build_summary(results_df, n_permutations, seed)

    return results_df, summary


def apply_fdr_correction(
    df: pd.DataFrame,
    pvalue_col: str,
    q_col: str
) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction.

    This controls the expected proportion of false discoveries.
    """
    df = df.copy()
    pvalues = df[pvalue_col].values
    n = len(pvalues)

    # Sort by p-value
    sorted_idx = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_idx]

    # Calculate BH adjusted p-values (q-values)
    q_values = np.zeros(n)
    cummin = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        bh_value = sorted_pvalues[i] * n / rank
        cummin = min(cummin, bh_value)
        q_values[sorted_idx[i]] = min(cummin, 1.0)

    df[q_col] = q_values
    return df


def build_summary(
    results_df: pd.DataFrame,
    n_permutations: int,
    seed: int
) -> Dict[str, Any]:
    """Build summary statistics for the hypothesis test."""

    # Count discoveries at different FDR thresholds
    shannon_discoveries_005 = (results_df["shannon_fdr_q"] <= 0.05).sum()
    shannon_discoveries_010 = (results_df["shannon_fdr_q"] <= 0.10).sum()
    simpson_discoveries_005 = (results_df["simpson_fdr_q"] <= 0.05).sum()
    simpson_discoveries_010 = (results_df["simpson_fdr_q"] <= 0.10).sum()

    # Top candidates (high diversity relative to null)
    top_shannon = results_df.nsmallest(10, "shannon_pvalue")[
        ["recclass", "count", "shannon_obs", "shannon_null_mean",
         "shannon_zscore", "shannon_pvalue", "shannon_fdr_q"]
    ].to_dict(orient="records")

    top_simpson = results_df.nsmallest(10, "simpson_pvalue")[
        ["recclass", "count", "simpson_obs", "simpson_null_mean",
         "simpson_zscore", "simpson_pvalue", "simpson_fdr_q"]
    ].to_dict(orient="records")

    # Classes with unexpectedly HIGH diversity (positive effect, low p-value)
    anomalous_high = results_df[
        (results_df["shannon_fdr_q"] <= 0.10) &
        (results_df["shannon_effect"] > 0) &
        (results_df["shannon_zscore"] >= 2)
    ].sort_values("shannon_zscore", ascending=False)

    return {
        "experiment": "O-Δ1",
        "hypothesis": "Diversity scales proportionally with sample size",
        "null_model": "Names randomly assigned to classes, preserving class sizes",
        "timestamp_utc": datetime.utcnow().isoformat(),
        "parameters": {
            "n_permutations": n_permutations,
            "seed": seed,
            "n_classes": len(results_df),
            "total_samples": results_df["count"].sum(),
        },
        "discoveries": {
            "shannon_fdr_005": int(shannon_discoveries_005),
            "shannon_fdr_010": int(shannon_discoveries_010),
            "simpson_fdr_005": int(simpson_discoveries_005),
            "simpson_fdr_010": int(simpson_discoveries_010),
        },
        "top_candidates_shannon": top_shannon,
        "top_candidates_simpson": top_simpson,
        "anomalous_high_diversity": anomalous_high[
            ["recclass", "count", "shannon_obs", "shannon_null_mean",
             "shannon_zscore", "shannon_fdr_q"]
        ].to_dict(orient="records") if len(anomalous_high) > 0 else [],
    }


def generate_hypothesis_plot(
    results_df: pd.DataFrame,
    output_path: Path
) -> str:
    """
    Generate visualization: observed vs null mean with significance markers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Shannon plot
    ax1 = axes[0]

    # Color by significance
    significant = results_df["shannon_fdr_q"] <= 0.10

    ax1.scatter(
        results_df.loc[~significant, "shannon_null_mean"],
        results_df.loc[~significant, "shannon_obs"],
        alpha=0.3, s=20, c="gray", label="Not significant"
    )
    ax1.scatter(
        results_df.loc[significant, "shannon_null_mean"],
        results_df.loc[significant, "shannon_obs"],
        alpha=0.8, s=40, c="red", label="FDR q ≤ 0.10"
    )

    # Identity line
    lims = [0, max(results_df["shannon_obs"].max(), results_df["shannon_null_mean"].max()) * 1.1]
    ax1.plot(lims, lims, 'k--', alpha=0.5, label="y = x (no deviation)")

    ax1.set_xlabel("Shannon (null mean)")
    ax1.set_ylabel("Shannon (observed)")
    ax1.set_title("Experiment O-Δ1: Shannon Entropy")
    ax1.legend(loc="lower right")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)

    # Simpson plot
    ax2 = axes[1]

    significant_s = results_df["simpson_fdr_q"] <= 0.10

    ax2.scatter(
        results_df.loc[~significant_s, "simpson_null_mean"],
        results_df.loc[~significant_s, "simpson_obs"],
        alpha=0.3, s=20, c="gray", label="Not significant"
    )
    ax2.scatter(
        results_df.loc[significant_s, "simpson_null_mean"],
        results_df.loc[significant_s, "simpson_obs"],
        alpha=0.8, s=40, c="red", label="FDR q ≤ 0.10"
    )

    lims_s = [0, 1.05]
    ax2.plot(lims_s, lims_s, 'k--', alpha=0.5, label="y = x (no deviation)")

    ax2.set_xlabel("Simpson (null mean)")
    ax2.set_ylabel("Simpson (observed)")
    ax2.set_title("Experiment O-Δ1: Simpson Index")
    ax2.legend(loc="lower right")
    ax2.set_xlim(lims_s)
    ax2.set_ylim(lims_s)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def run_experiment_o_delta_1(
    n_permutations: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run the complete O-Δ1 experiment and save results.

    Returns paths to generated files.
    """
    # Load data
    df = pd.read_parquet(DATA_PROCESSED / "meteorites.parquet")

    # Ensure required columns
    if "name" not in df.columns or "recclass" not in df.columns:
        raise ValueError("Dataset must have 'name' and 'recclass' columns")

    # Filter to valid data
    df = df.dropna(subset=["name", "recclass"])

    print(f"Running O-Δ1 with {n_permutations} permutations (seed={seed})...")

    # Run permutation test
    results_df, summary = run_permutation_test(
        df,
        n_permutations=n_permutations,
        seed=seed,
        progress_callback=lambda i, n: print(f"  Permutation {i}/{n}...") if i % 500 == 0 else None
    )

    # Save results
    REPORTS.mkdir(parents=True, exist_ok=True)

    csv_path = REPORTS / "hypothesis_O-Δ1_results.csv"
    results_df.to_csv(csv_path, index=False)

    json_path = REPORTS / "hypothesis_O-Δ1_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    plot_path = REPORTS / "hypothesis_O-Δ1_plot.png"
    generate_hypothesis_plot(results_df, plot_path)

    print(f"  Shannon discoveries (FDR≤0.05): {summary['discoveries']['shannon_fdr_005']}")
    print(f"  Simpson discoveries (FDR≤0.05): {summary['discoveries']['simpson_fdr_005']}")

    return {
        "csv": str(csv_path),
        "summary": str(json_path),
        "plot": str(plot_path),
        "discoveries": summary["discoveries"],
        "top_anomalous": summary["anomalous_high_diversity"][:5] if summary["anomalous_high_diversity"] else []
    }


if __name__ == "__main__":
    result = run_experiment_o_delta_1(n_permutations=1000, seed=42)
    print("\nExperiment complete:")
    print(f"  CSV: {result['csv']}")
    print(f"  Summary: {result['summary']}")
    print(f"  Plot: {result['plot']}")
