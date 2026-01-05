"""
O-Δ8: Sample Size Threshold for Structure Emergence

Key question from O-Δ7: Null-5 (balanced subsampling) destroys all structure.
Is the "structure" in L6, H6, etc. a sample size artifact?

Approach:
1. Vary subsample_size from 30 to 500
2. For each size, run balanced subsampling test
3. Track z-scores and significance
4. Identify the critical threshold where structure emerges

This reveals whether structure is:
- Real: appears at small N and persists
- Artifact: only appears at large N
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from originmap.config import DATA_PROCESSED, REPORTS
from .stats_robust import coefficient_of_variation, variance_of_log, mad_ratio


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


STAT_FUNCTIONS = {
    "cv": coefficient_of_variation,
    "varlog": variance_of_log,
    "mad": mad_ratio,
}


@dataclass
class ThresholdResult:
    """Result for a single class at varying sample sizes."""
    recclass: str
    full_n: int
    sample_sizes: List[int]
    z_scores: Dict[str, List[float]]  # stat -> [z at each sample size]
    p_values: Dict[str, List[float]]  # stat -> [p at each sample size]
    significant: Dict[str, List[bool]]  # stat -> [sig at each sample size]
    threshold_n: Dict[str, Optional[int]]  # stat -> first N where significant

    def to_dict(self) -> Dict:
        return {
            "recclass": self.recclass,
            "full_n": self.full_n,
            "sample_sizes": self.sample_sizes,
            "z_scores": self.z_scores,
            "p_values": self.p_values,
            "significant": self.significant,
            "threshold_n": self.threshold_n,
        }


def balanced_permutation_test(
    df: pd.DataFrame,
    target_class: str,
    stat_fn,
    subsample_size: int,
    n_permutations: int = 200,
    n_subsamples: int = 50,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Run balanced subsampling permutation test.

    1. Subsample target class to fixed size
    2. Compute statistic (average over multiple subsamples)
    3. Permute class labels with mass-bin stratification
    4. Compute null distribution

    Returns:
        observed: observed statistic (averaged)
        z_score: z-score vs null
        p_value: two-tailed p-value
    """
    rng = np.random.default_rng(seed)

    # Get target class data
    class_data = df[df["recclass"] == target_class]["mass"].values
    full_n = len(class_data)

    if full_n < subsample_size:
        return np.nan, np.nan, np.nan

    # Observed: average over subsamples
    obs_values = []
    for _ in range(n_subsamples):
        sample = rng.choice(class_data, size=subsample_size, replace=False)
        obs_values.append(stat_fn(sample))

    obs_values = np.array(obs_values)
    obs_values = obs_values[~np.isnan(obs_values)]

    if len(obs_values) < 10:
        return np.nan, np.nan, np.nan

    observed = np.mean(obs_values)

    # Create mass bins for stratification
    all_masses = df["mass"].values
    n_bins = 10
    bin_edges = np.percentile(all_masses, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1
    df_copy = df.copy()
    df_copy["mass_bin"] = np.digitize(df_copy["mass"], bin_edges) - 1

    # Null distribution
    null_values = []

    for _ in range(n_permutations):
        # Permute class labels within mass bins
        df_perm = df_copy.copy()
        for bin_val in df_perm["mass_bin"].unique():
            mask = df_perm["mass_bin"] == bin_val
            df_perm.loc[mask, "recclass"] = rng.permutation(
                df_perm.loc[mask, "recclass"].values
            )

        # Get permuted target class
        perm_class_data = df_perm[df_perm["recclass"] == target_class]["mass"].values

        if len(perm_class_data) >= subsample_size:
            sample = rng.choice(perm_class_data, size=subsample_size, replace=False)
            null_values.append(stat_fn(sample))

    null_values = np.array(null_values)
    null_values = null_values[~np.isnan(null_values)]

    if len(null_values) < 10:
        return observed, np.nan, np.nan

    null_mean = np.mean(null_values)
    null_std = np.std(null_values, ddof=1)

    if null_std == 0:
        z_score = 0.0
    else:
        z_score = (observed - null_mean) / null_std

    # Two-tailed p-value
    p_value = 2 * min(
        np.mean(null_values <= observed),
        np.mean(null_values >= observed)
    )

    return observed, z_score, p_value


def run_threshold_sweep(
    target_classes: List[str],
    sample_sizes: List[int],
    stat_names: List[str] = ["cv", "varlog", "mad"],
    n_permutations: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> List[ThresholdResult]:
    """
    Run threshold sweep for multiple classes and sample sizes.
    """
    # Load data
    df = pd.read_parquet(DATA_PROCESSED / "meteorites.parquet")
    df = df.dropna(subset=["mass", "recclass"])
    df = df[df["mass"] > 0].reset_index(drop=True)

    results = []

    for recclass in target_classes:
        class_n = (df["recclass"] == recclass).sum()

        if class_n < min(sample_sizes):
            print(f"  {recclass}: insufficient data (n={class_n})")
            continue

        z_scores = {stat: [] for stat in stat_names}
        p_values = {stat: [] for stat in stat_names}
        significant = {stat: [] for stat in stat_names}

        valid_sizes = [s for s in sample_sizes if s <= class_n]

        print(f"  {recclass} (n={class_n}): testing {len(valid_sizes)} sizes...", end=" ", flush=True)

        for subsample_size in valid_sizes:
            for stat_name in stat_names:
                stat_fn = STAT_FUNCTIONS[stat_name]

                _, z, p = balanced_permutation_test(
                    df, recclass, stat_fn, subsample_size,
                    n_permutations=n_permutations,
                    seed=seed
                )

                z_scores[stat_name].append(z)
                p_values[stat_name].append(p)
                # Significant if p <= alpha AND z < 0 (tighter than expected)
                is_sig = (not np.isnan(p)) and (p <= alpha) and (z < 0)
                significant[stat_name].append(is_sig)

        # Find threshold: first N where significant
        threshold_n = {}
        for stat_name in stat_names:
            sig_list = significant[stat_name]
            threshold_n[stat_name] = None
            for i, is_sig in enumerate(sig_list):
                if is_sig:
                    threshold_n[stat_name] = valid_sizes[i]
                    break

        results.append(ThresholdResult(
            recclass=recclass,
            full_n=class_n,
            sample_sizes=valid_sizes,
            z_scores=z_scores,
            p_values=p_values,
            significant=significant,
            threshold_n=threshold_n,
        ))

        print("done")

    return results


def run_o_delta_8(
    sample_sizes: List[int] = [30, 50, 75, 100, 150, 200, 300, 500],
    target_classes: Optional[List[str]] = None,
    stat_names: List[str] = ["cv", "varlog", "mad"],
    n_permutations: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Run O-Δ8 experiment: Sample size threshold analysis.
    """
    print("=" * 70)
    print("O-Δ8: Sample Size Threshold for Structure Emergence")
    print("=" * 70)
    print(f"Sample sizes: {sample_sizes}")
    print(f"Statistics: {stat_names}")
    print(f"Permutations: {n_permutations}, Alpha: {alpha}")
    print()

    # Default target classes: those from Regime 3 (high stability) in O-Δ7
    if target_classes is None:
        target_classes = [
            "L6", "H6", "H4", "Ureilite", "Eucrite-pmict",
            "H5", "L5", "LL6", "LL5",  # Also test some from other regimes
        ]

    print(f"Target classes: {target_classes}")
    print()

    print("-" * 70)
    print("Running threshold sweep...")
    print("-" * 70)

    results = run_threshold_sweep(
        target_classes=target_classes,
        sample_sizes=sample_sizes,
        stat_names=stat_names,
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha,
    )

    # Analyze results
    print()
    print("-" * 70)
    print("THRESHOLD ANALYSIS")
    print("-" * 70)

    for r in results:
        print(f"\n  {r.recclass} (full n={r.full_n}):")
        for stat in stat_names:
            thresh = r.threshold_n.get(stat)
            if thresh:
                print(f"    {stat}: structure emerges at N≥{thresh}")
            else:
                # Check if ever significant
                if any(r.significant.get(stat, [])):
                    last_sig = None
                    for i, s in enumerate(r.significant[stat]):
                        if s:
                            last_sig = r.sample_sizes[i]
                    print(f"    {stat}: structure present at N={last_sig}")
                else:
                    print(f"    {stat}: NO structure at any tested N")

    # Summary statistics
    print()
    print("-" * 70)
    print("EMERGENCE PATTERN SUMMARY")
    print("-" * 70)

    for stat in stat_names:
        thresholds = [r.threshold_n.get(stat) for r in results if r.threshold_n.get(stat)]
        never_sig = sum(1 for r in results if not any(r.significant.get(stat, [])))

        if thresholds:
            print(f"\n  {stat}:")
            print(f"    Classes with structure: {len(thresholds)}/{len(results)}")
            print(f"    Min threshold: N={min(thresholds)}")
            print(f"    Max threshold: N={max(thresholds)}")
            print(f"    Median threshold: N={int(np.median(thresholds))}")
        else:
            print(f"\n  {stat}: No class shows structure")

        if never_sig:
            print(f"    Never significant: {never_sig} classes")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "sample_sizes": sample_sizes,
            "target_classes": target_classes,
            "stat_names": stat_names,
            "n_permutations": n_permutations,
            "seed": seed,
            "alpha": alpha,
        },
        "results": [r.to_dict() for r in results],
    }


def generate_outputs(results: Dict[str, Any], output_dir: Path = REPORTS) -> Dict[str, str]:
    """Generate O-Δ8 outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    # 1. CSV: Threshold summary
    csv_data = []
    for r in results["results"]:
        for stat in results["parameters"]["stat_names"]:
            csv_data.append({
                "recclass": r["recclass"],
                "full_n": r["full_n"],
                "stat": stat,
                "threshold_n": r["threshold_n"].get(stat),
                "ever_significant": any(r["significant"].get(stat, [])),
            })

    csv_df = pd.DataFrame(csv_data)
    csv_path = output_dir / "O-D8_threshold_summary.csv"
    csv_df.to_csv(csv_path, index=False)
    files["csv"] = str(csv_path)

    # 2. JSON: Full results
    json_path = output_dir / "O-D8_threshold_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    files["json"] = str(json_path)

    # 3. Detailed CSV: All z-scores by sample size
    detail_data = []
    for r in results["results"]:
        for i, size in enumerate(r["sample_sizes"]):
            row = {
                "recclass": r["recclass"],
                "sample_size": size,
            }
            for stat in results["parameters"]["stat_names"]:
                z_list = r["z_scores"].get(stat, [])
                p_list = r["p_values"].get(stat, [])
                s_list = r["significant"].get(stat, [])

                row[f"z_{stat}"] = z_list[i] if i < len(z_list) else np.nan
                row[f"p_{stat}"] = p_list[i] if i < len(p_list) else np.nan
                row[f"sig_{stat}"] = s_list[i] if i < len(s_list) else False

            detail_data.append(row)

    detail_df = pd.DataFrame(detail_data)
    detail_path = output_dir / "O-D8_threshold_detail.csv"
    detail_df.to_csv(detail_path, index=False)
    files["detail"] = str(detail_path)

    # 4. Plot: Z-score trajectories
    plot_path = output_dir / "O-D8_threshold_curves.png"
    generate_threshold_plot(results, plot_path)
    files["plot"] = str(plot_path)

    # 5. Plot: Heatmap of significance
    heatmap_path = output_dir / "O-D8_significance_heatmap.png"
    generate_significance_heatmap(results, heatmap_path)
    files["heatmap"] = str(heatmap_path)

    print()
    print("=" * 70)
    print("FILES GENERATED")
    print("=" * 70)
    for k, v in files.items():
        print(f"  {k}: {v}")

    return files


def generate_threshold_plot(results: Dict[str, Any], output_path: Path):
    """Generate z-score trajectory plot."""
    stat_names = results["parameters"]["stat_names"]
    n_stats = len(stat_names)

    fig, axes = plt.subplots(1, n_stats, figsize=(6 * n_stats, 6))
    if n_stats == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(results["results"])))

    for ax_idx, stat in enumerate(stat_names):
        ax = axes[ax_idx]

        for i, r in enumerate(results["results"]):
            sizes = r["sample_sizes"]
            z_scores = r["z_scores"].get(stat, [])

            if len(z_scores) > 0:
                ax.plot(sizes[:len(z_scores)], z_scores, 'o-',
                       color=colors[i], label=r["recclass"], alpha=0.7)

                # Mark threshold
                thresh = r["threshold_n"].get(stat)
                if thresh:
                    idx = sizes.index(thresh) if thresh in sizes else None
                    if idx is not None and idx < len(z_scores):
                        ax.scatter([thresh], [z_scores[idx]], s=100,
                                  color=colors[i], marker='*', zorder=5)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-2, color='red', linestyle='--', alpha=0.5, label='z=-2')
        ax.set_xlabel("Subsample Size (N)", fontsize=12)
        ax.set_ylabel("Z-score", fontsize=12)
        ax.set_title(f"{stat}: Z-score vs Sample Size\n(★ = emergence threshold)", fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("O-Δ8: Sample Size Threshold Analysis\n"
                "At what N does structure emerge?", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_significance_heatmap(results: Dict[str, Any], output_path: Path):
    """Generate significance heatmap."""
    stat_names = results["parameters"]["stat_names"]

    # Build matrix: rows = classes, cols = sample_sizes × stats
    classes = [r["recclass"] for r in results["results"]]
    all_sizes = sorted(set(s for r in results["results"] for s in r["sample_sizes"]))

    fig, axes = plt.subplots(1, len(stat_names), figsize=(5 * len(stat_names), 8))
    if len(stat_names) == 1:
        axes = [axes]

    for ax_idx, stat in enumerate(stat_names):
        ax = axes[ax_idx]

        # Build significance matrix
        matrix = np.zeros((len(classes), len(all_sizes)))
        matrix[:] = np.nan

        for i, r in enumerate(results["results"]):
            for j, size in enumerate(r["sample_sizes"]):
                if size in all_sizes:
                    col_idx = all_sizes.index(size)
                    sig_list = r["significant"].get(stat, [])
                    if j < len(sig_list):
                        matrix[i, col_idx] = 1 if sig_list[j] else 0

        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xticks(range(len(all_sizes)))
        ax.set_xticklabels(all_sizes, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel("Subsample Size", fontsize=11)
        ax.set_ylabel("Class", fontsize=11)
        ax.set_title(f"{stat}: Significance Map\n(green=significant, red=not)", fontsize=11)

        # Mark thresholds
        for i, r in enumerate(results["results"]):
            thresh = r["threshold_n"].get(stat)
            if thresh and thresh in all_sizes:
                col_idx = all_sizes.index(thresh)
                ax.scatter([col_idx], [i], s=50, marker='|', color='black', linewidths=2)

    plt.suptitle("O-Δ8: Significance Emergence Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_observation_md(results: Dict[str, Any], files: Dict[str, str], output_dir: Path) -> str:
    """Generate observation markdown."""
    date = datetime.now().strftime("%Y%m%d")
    md_path = output_dir / f"observation_O-D8_{date}.md"

    params = results["parameters"]
    stat_names = params["stat_names"]

    lines = [
        "# Observation O-Δ8: Sample Size Threshold Analysis",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Experiment**: O-Δ8 (Sample Size Threshold)",
        f"**Sample sizes tested**: {params['sample_sizes']}",
        f"**Permutations**: {params['n_permutations']}",
        f"**Alpha**: {params['alpha']}",
        "",
        "---",
        "",
        "## Question",
        "",
        "From O-Δ7: Null-5 (balanced subsampling at N=100) destroys all structure.",
        "Is the 'structure' in L6, H6, etc. a sample size artifact?",
        "",
        "**At what N does structure emerge?**",
        "",
        "---",
        "",
        "## Results by Statistic",
        "",
    ]

    for stat in stat_names:
        thresholds = []
        never_sig = []

        for r in results["results"]:
            thresh = r["threshold_n"].get(stat)
            if thresh:
                thresholds.append((r["recclass"], thresh))
            elif not any(r["significant"].get(stat, [])):
                never_sig.append(r["recclass"])

        lines.extend([
            f"### {stat.upper()}",
            "",
        ])

        if thresholds:
            lines.append("| Class | Threshold N |")
            lines.append("|-------|------------|")
            for recclass, thresh in sorted(thresholds, key=lambda x: x[1]):
                lines.append(f"| {recclass} | {thresh} |")
            lines.append("")

        if never_sig:
            lines.append(f"**Never significant**: {', '.join(never_sig)}")
            lines.append("")

    # Key finding
    all_thresholds = []
    for r in results["results"]:
        for stat in stat_names:
            thresh = r["threshold_n"].get(stat)
            if thresh:
                all_thresholds.append(thresh)

    if all_thresholds:
        min_thresh = min(all_thresholds)
        median_thresh = int(np.median(all_thresholds))

        lines.extend([
            "---",
            "",
            "## Key Finding",
            "",
            f"- **Minimum threshold**: N = {min_thresh}",
            f"- **Median threshold**: N = {median_thresh}",
            "",
        ])

        if min_thresh <= 50:
            lines.append("**Interpretation**: Structure appears at small N → likely REAL effect")
        elif min_thresh >= 200:
            lines.append("**Interpretation**: Structure only at large N → possible ARTIFACT")
        else:
            lines.append("**Interpretation**: Intermediate threshold → mixed evidence")
    else:
        lines.extend([
            "---",
            "",
            "## Key Finding",
            "",
            "**No class shows significant structure at any tested sample size.**",
            "",
            "This suggests the 'structure' observed in the full battery is entirely",
            "a sample size artifact.",
        ])

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
        "*Generated by ORIGINMAP O-Δ8 experiment*",
    ])

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return str(md_path)


def run_full_o_delta_8(
    sample_sizes: List[int] = [30, 50, 75, 100, 150, 200, 300, 500],
    target_classes: Optional[List[str]] = None,
    stat_names: List[str] = ["cv", "varlog", "mad"],
    n_permutations: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, str]:
    """
    Main entry point for O-Δ8 experiment.
    """
    results = run_o_delta_8(
        sample_sizes=sample_sizes,
        target_classes=target_classes,
        stat_names=stat_names,
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha,
    )

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
    print("O-Δ8 VERDICT")
    print("=" * 70)

    all_thresholds = []
    for r in results["results"]:
        for stat in stat_names:
            thresh = r["threshold_n"].get(stat)
            if thresh:
                all_thresholds.append((r["recclass"], stat, thresh))

    if all_thresholds:
        min_thresh = min(t[2] for t in all_thresholds)
        max_thresh = max(t[2] for t in all_thresholds)

        print(f"\n  Threshold range: N={min_thresh} to N={max_thresh}")
        print()
        print("  Early emergence (N≤50):")
        for rc, stat, thresh in sorted(all_thresholds, key=lambda x: x[2]):
            if thresh <= 50:
                print(f"    {rc} ({stat}): N={thresh}")

        print()
        if min_thresh <= 50:
            print("  → VERDICT: Structure is REAL (emerges at small N)")
        elif min_thresh >= 200:
            print("  → VERDICT: Structure is likely ARTIFACT (only at large N)")
        else:
            print("  → VERDICT: MIXED - some structure may be real, some artifact")
    else:
        print("\n  No class shows significant structure at any N.")
        print("  → VERDICT: All observed structure is sample size ARTIFACT")

    return files


if __name__ == "__main__":
    run_full_o_delta_8()
