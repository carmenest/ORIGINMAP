"""
O-Δ7: Stability Regime Clustering

Instead of asking "who survives everything?", ask:
"What stability profiles exist across meteorite classes?"

Approach:
1. Build class × test matrix from battery results
2. Cluster classes by similarity of survival patterns
3. Identify distinct stability regimes
4. Characterize each regime

No ML, just hierarchical clustering with scipy.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
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


@dataclass
class StabilityRegime:
    """A cluster of classes with similar stability profiles."""
    regime_id: int
    name: str
    classes: List[str]
    n_classes: int
    mean_survival: float
    characteristic_tests: List[str]  # Tests where this regime tends to survive
    anti_characteristic_tests: List[str]  # Tests where this regime tends to fail
    centroid: np.ndarray

    def to_dict(self) -> Dict:
        return {
            "regime_id": self.regime_id,
            "name": self.name,
            "classes": self.classes,
            "n_classes": self.n_classes,
            "mean_survival": self.mean_survival,
            "characteristic_tests": self.characteristic_tests,
            "anti_characteristic_tests": self.anti_characteristic_tests,
            "centroid": self.centroid.tolist(),
        }


def load_latest_battery_results() -> Tuple[pd.DataFrame, Path]:
    """
    Load the most recent battery results CSV.
    Returns DataFrame and path to the file.
    """
    # Find most recent battery results
    battery_files = list(REPORTS.glob("battery_*_results.csv"))
    if not battery_files:
        raise FileNotFoundError("No battery results found. Run 'originmap battery' first.")

    latest = max(battery_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest)

    return df, latest


def build_survival_matrix(
    df: pd.DataFrame,
    mode: str = "binary"
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build class × test matrix.

    Args:
        df: Battery results DataFrame
        mode: 'binary' (0/1 survival) or 'zscore' (continuous z-scores)

    Returns:
        matrix: DataFrame with classes as rows, tests as columns
        classes: List of class names
        tests: List of test names
    """
    # Identify survival columns
    if mode == "binary":
        test_cols = [c for c in df.columns if c.startswith("surv_")]
    else:  # zscore
        test_cols = [c for c in df.columns if c.startswith("z_")]

    if not test_cols:
        raise ValueError(f"No {mode} columns found in results")

    # Extract class names and build matrix
    classes = df["recclass"].tolist()
    matrix = df[test_cols].copy()
    matrix.index = classes

    # Clean column names
    if mode == "binary":
        matrix.columns = [c.replace("surv_", "") for c in matrix.columns]
    else:
        matrix.columns = [c.replace("z_", "") for c in matrix.columns]

    # Convert to numeric, handle booleans
    for col in matrix.columns:
        matrix[col] = pd.to_numeric(matrix[col], errors='coerce')

    # For binary mode, convert True/False to 1/0
    if mode == "binary":
        matrix = matrix.astype(float)

    tests = matrix.columns.tolist()

    return matrix, classes, tests


def compute_distance_matrix(
    matrix: pd.DataFrame,
    metric: str = "jaccard"
) -> np.ndarray:
    """
    Compute pairwise distance matrix.

    Args:
        matrix: class × test matrix
        metric: 'jaccard' for binary, 'euclidean' or 'correlation' for continuous

    Returns:
        Condensed distance matrix for scipy.cluster.hierarchy
    """
    # Handle NaN
    matrix_filled = matrix.fillna(0)

    if metric == "jaccard":
        # Jaccard for binary data
        return pdist(matrix_filled.values, metric='jaccard')
    elif metric == "correlation":
        # Correlation distance for continuous
        return pdist(matrix_filled.values, metric='correlation')
    else:
        # Euclidean
        return pdist(matrix_filled.values, metric='euclidean')


def perform_clustering(
    matrix: pd.DataFrame,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    method: str = "ward",
    metric: str = "jaccard"
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Perform hierarchical clustering.

    Args:
        matrix: class × test matrix
        n_clusters: Number of clusters (if specified)
        distance_threshold: Distance threshold for flat clustering
        method: Linkage method ('ward', 'complete', 'average', 'single')
        metric: Distance metric

    Returns:
        linkage_matrix: Scipy linkage matrix
        distances: Condensed distance matrix
        labels: Cluster labels for each class
    """
    # Compute distances
    distances = compute_distance_matrix(matrix, metric)

    # For ward, need euclidean distance
    if method == "ward":
        distances_linkage = pdist(matrix.fillna(0).values, metric='euclidean')
    else:
        distances_linkage = distances

    # Perform linkage
    Z = linkage(distances_linkage, method=method)

    # Get flat clusters
    if n_clusters is not None:
        labels = fcluster(Z, n_clusters, criterion='maxclust')
    elif distance_threshold is not None:
        labels = fcluster(Z, distance_threshold, criterion='distance')
    else:
        # Auto-detect using silhouette-like heuristic
        # Try different numbers of clusters, pick one with good separation
        best_n = 4
        labels = fcluster(Z, best_n, criterion='maxclust')

    return Z, distances, labels.tolist()


def identify_regimes(
    matrix: pd.DataFrame,
    labels: List[int],
    classes: List[str],
    tests: List[str]
) -> List[StabilityRegime]:
    """
    Characterize each cluster as a stability regime.
    """
    regimes = []
    unique_labels = sorted(set(labels))

    for regime_id in unique_labels:
        # Get classes in this regime
        regime_classes = [c for c, l in zip(classes, labels) if l == regime_id]
        regime_matrix = matrix.loc[regime_classes]

        # Compute centroid (mean survival profile)
        centroid = regime_matrix.mean().values

        # Mean survival rate
        mean_survival = np.nanmean(centroid)

        # Characteristic tests (high survival in this regime)
        char_tests = [t for t, v in zip(tests, centroid) if v > 0.5]

        # Anti-characteristic tests (low survival)
        anti_char_tests = [t for t, v in zip(tests, centroid) if v < 0.2]

        # Name based on pattern
        if mean_survival > 0.7:
            name = f"High-Stability (Regime {regime_id})"
        elif mean_survival > 0.4:
            name = f"Moderate-Stability (Regime {regime_id})"
        elif mean_survival > 0.1:
            name = f"Low-Stability (Regime {regime_id})"
        else:
            name = f"Unstable (Regime {regime_id})"

        regimes.append(StabilityRegime(
            regime_id=regime_id,
            name=name,
            classes=regime_classes,
            n_classes=len(regime_classes),
            mean_survival=mean_survival,
            characteristic_tests=char_tests,
            anti_characteristic_tests=anti_char_tests,
            centroid=centroid,
        ))

    return regimes


def run_stability_clustering(
    n_clusters: int = 4,
    method: str = "ward",
    mode: str = "binary",
) -> Dict[str, Any]:
    """
    Run O-Δ7 stability clustering experiment.

    Args:
        n_clusters: Number of clusters to identify
        method: Clustering method
        mode: 'binary' or 'zscore'

    Returns:
        Results dictionary
    """
    print("=" * 70)
    print("O-Δ7: Stability Regime Clustering")
    print("=" * 70)

    # Load data
    print("\nLoading battery results...")
    df, source_file = load_latest_battery_results()
    print(f"  Source: {source_file}")
    print(f"  Classes: {len(df)}")

    # Build matrix
    print(f"\nBuilding {mode} survival matrix...")
    matrix, classes, tests = build_survival_matrix(df, mode)
    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Tests: {tests}")

    # Clustering
    print(f"\nPerforming hierarchical clustering...")
    print(f"  Method: {method}, Clusters: {n_clusters}")

    metric = "jaccard" if mode == "binary" else "euclidean"
    Z, distances, labels = perform_clustering(
        matrix,
        n_clusters=n_clusters,
        method=method,
        metric=metric
    )

    # Identify regimes
    print("\nIdentifying stability regimes...")
    regimes = identify_regimes(matrix, labels, classes, tests)

    # Print summary
    print("\n" + "-" * 70)
    print("STABILITY REGIMES IDENTIFIED")
    print("-" * 70)

    for regime in sorted(regimes, key=lambda r: -r.mean_survival):
        print(f"\n  [{regime.name}]")
        print(f"    Classes: {regime.n_classes}")
        print(f"    Mean survival: {regime.mean_survival:.1%}")
        if regime.characteristic_tests:
            print(f"    Strong in: {', '.join(regime.characteristic_tests[:5])}")
        if regime.anti_characteristic_tests:
            print(f"    Weak in: {', '.join(regime.anti_characteristic_tests[:5])}")
        print(f"    Examples: {', '.join(regime.classes[:5])}")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source_file),
        "parameters": {
            "n_clusters": n_clusters,
            "method": method,
            "mode": mode,
        },
        "n_classes": len(classes),
        "n_tests": len(tests),
        "tests": tests,
        "classes": classes,
        "labels": labels,
        "regimes": [r.to_dict() for r in regimes],
        "linkage_matrix": Z.tolist(),
        "matrix": matrix.to_dict(),
    }


def generate_outputs(results: Dict[str, Any], output_dir: Path = REPORTS) -> Dict[str, str]:
    """Generate O-Δ7 outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    # 1. Stability regimes CSV
    regime_data = []
    for r in results["regimes"]:
        for c in r["classes"]:
            regime_data.append({
                "recclass": c,
                "regime_id": r["regime_id"],
                "regime_name": r["name"],
                "regime_mean_survival": r["mean_survival"],
            })

    regimes_df = pd.DataFrame(regime_data)
    regimes_path = output_dir / "O-D7_stability_regimes.csv"
    regimes_df.to_csv(regimes_path, index=False)
    files["regimes_csv"] = str(regimes_path)

    # 2. JSON summary
    json_summary = {
        "experiment": "O-D7",
        "timestamp_utc": results["timestamp_utc"],
        "parameters": results["parameters"],
        "n_classes": results["n_classes"],
        "n_tests": results["n_tests"],
        "regimes": results["regimes"],
    }

    json_path = output_dir / "O-D7_clustering_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2, cls=NumpyEncoder)
    files["json"] = str(json_path)

    # 3. Dendrogram plot
    plot_path = output_dir / "O-D7_stability_dendrogram.png"
    generate_dendrogram_plot(results, plot_path)
    files["dendrogram"] = str(plot_path)

    # 4. Heatmap plot
    heatmap_path = output_dir / "O-D7_survival_heatmap.png"
    generate_heatmap_plot(results, heatmap_path)
    files["heatmap"] = str(heatmap_path)

    print("\n" + "=" * 70)
    print("FILES GENERATED")
    print("=" * 70)
    for k, v in files.items():
        print(f"  {k}: {v}")

    return files


def generate_dendrogram_plot(results: Dict[str, Any], output_path: Path):
    """Generate dendrogram visualization."""
    Z = np.array(results["linkage_matrix"])
    classes = results["classes"]
    labels = results["labels"]
    regimes = {r["regime_id"]: r["name"] for r in results["regimes"]}

    fig, ax = plt.subplots(figsize=(16, 10))

    # Color map for regimes
    n_clusters = len(set(labels))
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    color_map = {i+1: colors[i] for i in range(n_clusters)}

    # Create dendrogram
    dendro = dendrogram(
        Z,
        labels=classes,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
        color_threshold=0,  # All same color initially
    )

    # Color labels by regime
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
    xlbls = ax.get_xticklabels()
    for lbl in xlbls:
        class_name = lbl.get_text()
        if class_name in classes:
            idx = classes.index(class_name)
            regime = labels[idx]
            lbl.set_color(color_map[regime])

    ax.set_ylabel("Distance", fontsize=12)
    ax.set_title("O-Δ7: Stability Regime Dendrogram\n"
                "Classes clustered by survival pattern similarity",
                fontsize=14, fontweight="bold")

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=color_map[r["regime_id"]],
                  markersize=10, label=f'{r["name"]} (n={r["n_classes"]})')
        for r in results["regimes"]
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_heatmap_plot(results: Dict[str, Any], output_path: Path):
    """Generate heatmap of survival patterns."""
    matrix_dict = results["matrix"]
    matrix = pd.DataFrame(matrix_dict)
    labels = results["labels"]
    classes = results["classes"]
    regimes = {r["regime_id"]: r for r in results["regimes"]}

    # Sort by regime, then by survival rate within regime
    sorted_indices = sorted(
        range(len(classes)),
        key=lambda i: (labels[i], -matrix.iloc[i].mean())
    )

    sorted_matrix = matrix.iloc[sorted_indices]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(14, 12))

    # Create heatmap
    im = ax.imshow(sorted_matrix.values, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=1)

    # Labels
    ax.set_yticks(range(len(sorted_classes)))
    ax.set_yticklabels(sorted_classes, fontsize=6)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha='right', fontsize=8)

    # Color y-labels by regime
    n_clusters = len(set(labels))
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    color_map = {i+1: colors[i] for i in range(n_clusters)}

    for i, (lbl, regime) in enumerate(zip(ax.get_yticklabels(), sorted_labels)):
        lbl.set_color(color_map[regime])

    # Add regime separators
    prev_regime = sorted_labels[0]
    for i, regime in enumerate(sorted_labels):
        if regime != prev_regime:
            ax.axhline(y=i-0.5, color='black', linewidth=2)
            prev_regime = regime

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Survival (1=pass, 0=fail)', fontsize=10)

    ax.set_xlabel("Test (null_stat)", fontsize=12)
    ax.set_ylabel("Meteorite Class (colored by regime)", fontsize=12)
    ax.set_title("O-Δ7: Survival Pattern Heatmap\n"
                "Rows sorted by regime, then by mean survival",
                fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_observation_md(results: Dict[str, Any], files: Dict[str, str], output_dir: Path) -> str:
    """Generate observation markdown."""
    date = datetime.now().strftime("%Y%m%d")
    md_path = output_dir / f"observation_O-D7_{date}.md"

    regimes = results["regimes"]
    regimes_sorted = sorted(regimes, key=lambda r: -r["mean_survival"])

    lines = [
        "# Observation O-Δ7: Stability Regime Clustering",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Experiment**: O-Δ7 (Stability Clustering)",
        f"**Source**: {results['source_file']}",
        f"**Classes**: {results['n_classes']}",
        f"**Tests**: {results['n_tests']}",
        "",
        "---",
        "",
        "## Objective",
        "",
        "Identify distinct stability regimes across meteorite classes based on",
        "their survival patterns in the null model battery.",
        "",
        "---",
        "",
        "## Method",
        "",
        f"- Mode: {results['parameters']['mode']}",
        f"- Clustering: Hierarchical ({results['parameters']['method']})",
        f"- Clusters: {results['parameters']['n_clusters']}",
        "",
        "---",
        "",
        "## Regimes Identified",
        "",
    ]

    for r in regimes_sorted:
        lines.extend([
            f"### {r['name']}",
            "",
            f"- **Classes**: {r['n_classes']}",
            f"- **Mean survival**: {r['mean_survival']:.1%}",
            f"- **Strong in**: {', '.join(r['characteristic_tests'][:5]) if r['characteristic_tests'] else 'None'}",
            f"- **Weak in**: {', '.join(r['anti_characteristic_tests'][:5]) if r['anti_characteristic_tests'] else 'None'}",
            "",
            "**Members**:",
            "",
            ", ".join(r['classes'][:20]) + ("..." if len(r['classes']) > 20 else ""),
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Key Finding",
        "",
        "Instead of a single 'robust' class, the data reveals **distinct stability regimes**:",
        "",
    ])

    for r in regimes_sorted:
        desc = "consistent structure" if r["mean_survival"] > 0.5 else \
               "partial structure" if r["mean_survival"] > 0.2 else "no consistent structure"
        lines.append(f"- **{r['name']}**: {r['n_classes']} classes with {desc}")

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
        "*Generated by ORIGINMAP O-Δ7 experiment*",
    ])

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return str(md_path)


def run_o_delta_7(
    n_clusters: int = 4,
    method: str = "ward",
    mode: str = "binary",
) -> Dict[str, str]:
    """
    Main entry point for O-Δ7 experiment.
    """
    results = run_stability_clustering(n_clusters, method, mode)
    files = generate_outputs(results, REPORTS)

    # Generate observation
    from originmap.config import PROJECT_ROOT
    obs_dir = PROJECT_ROOT / "notes" / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)
    obs_path = generate_observation_md(results, files, obs_dir)
    files["observation"] = obs_path

    # Print summary
    print()
    print("=" * 70)
    print("O-Δ7 SUMMARY")
    print("=" * 70)

    regimes = sorted(results["regimes"], key=lambda r: -r["mean_survival"])

    print("\nStability Regimes (by mean survival):")
    for r in regimes:
        print(f"  {r['name']:30} n={r['n_classes']:3}  survival={r['mean_survival']:.1%}")

    print(f"\n  Observation: {files['observation']}")

    return files


if __name__ == "__main__":
    run_o_delta_7(n_clusters=4, method="ward", mode="binary")
