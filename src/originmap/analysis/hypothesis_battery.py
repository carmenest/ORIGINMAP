"""
Hypothesis Battery Orchestrator.

Runs comprehensive null model battery tests with:
- Multiple null models (Null-1 through Null-5)
- Multiple statistics (CV, varlog, MAD/median)
- Multiple bin configurations
- Bootstrap confidence intervals
- Auto-generated Battery Report

Usage:
    python -m originmap.cli hypothesis battery --null 1-5 --bins 8,10,12,20 --n 500
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from originmap.config import DATA_PROCESSED, REPORTS
from .null_models import (
    NULL_MODELS, get_null_model, parse_null_range,
    Null5Balanced, NullResult
)
from .stats_robust import (
    STAT_FUNCTIONS, compute_all_stats, bootstrap_stats
)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
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
class ClassResult:
    """Results for a single meteorite class."""
    recclass: str
    count: int
    stats_observed: Dict[str, float]
    stats_bootstrap: Dict[str, tuple]
    null_results: Dict[str, Dict[str, NullResult]]  # null_name -> stat_name -> result
    survival: Dict[str, bool]
    total_survived: int


@dataclass
class BatteryReport:
    """Complete battery report."""
    timestamp_utc: str
    parameters: Dict[str, Any]
    null_models_used: List[str]
    stats_used: List[str]
    bin_configs: List[int]
    total_classes_tested: int
    class_results: Dict[str, ClassResult]
    survival_summary: Dict[str, int]
    robust_candidates: List[str]
    sensitivity_results: Dict[int, Dict[str, List[str]]]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp_utc": self.timestamp_utc,
            "parameters": self.parameters,
            "null_models_used": self.null_models_used,
            "stats_used": self.stats_used,
            "bin_configs": self.bin_configs,
            "total_classes_tested": self.total_classes_tested,
            "survival_summary": self.survival_summary,
            "robust_candidates": self.robust_candidates,
            "class_results": {
                rc: {
                    "recclass": cr.recclass,
                    "count": cr.count,
                    "stats_observed": cr.stats_observed,
                    "survival": cr.survival,
                    "total_survived": cr.total_survived,
                    "null_results": {
                        null_name: {
                            stat_name: nr.to_dict()
                            for stat_name, nr in stat_results.items()
                        }
                        for null_name, stat_results in cr.null_results.items()
                    }
                }
                for rc, cr in self.class_results.items()
            },
            "sensitivity_results": self.sensitivity_results,
        }


def apply_fdr(pvalues: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    valid_mask = ~np.isnan(pvalues)
    n_valid = valid_mask.sum()

    if n_valid == 0:
        return np.full(len(pvalues), np.nan)

    valid_pvalues = pvalues[valid_mask]
    sorted_idx = np.argsort(valid_pvalues)
    sorted_pvalues = valid_pvalues[sorted_idx]

    q_valid = np.zeros(n_valid)
    cummin = 1.0

    for i in range(n_valid - 1, -1, -1):
        rank = i + 1
        bh = sorted_pvalues[i] * n_valid / rank
        cummin = min(cummin, bh)
        q_valid[sorted_idx[i]] = min(cummin, 1.0)

    q_values = np.full(len(pvalues), np.nan)
    q_values[valid_mask] = q_valid
    return q_values


def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare meteorite data."""
    df = pd.read_parquet(DATA_PROCESSED / "meteorites.parquet")
    df = df.dropna(subset=["mass", "recclass"])
    df = df[df["mass"] > 0].reset_index(drop=True)
    return df


def run_battery(
    null_spec: str = "1-5",
    stat_names: List[str] = ["cv", "varlog", "mad_ratio"],
    bin_configs: List[int] = [10],
    n_permutations: int = 500,
    n_bootstrap: int = 200,
    seed: int = 42,
    min_samples: int = 30,
    fdr_threshold: float = 0.10,
    subsample_size: int = 100,
) -> BatteryReport:
    """
    Run comprehensive null model battery.

    Args:
        null_spec: Null models to run (e.g., "1-5", "2,4", "1-3,5")
        stat_names: Statistics to compute
        bin_configs: List of bin counts for sensitivity analysis
        n_permutations: Number of permutations per test
        n_bootstrap: Number of bootstrap samples
        seed: Random seed
        min_samples: Minimum samples per class
        fdr_threshold: FDR threshold for significance
        subsample_size: Size for balanced subsampling (Null-5)

    Returns:
        BatteryReport with complete results
    """
    print("=" * 70)
    print("ORIGINMAP — Comprehensive Null Model Battery")
    print("=" * 70)

    null_models = parse_null_range(null_spec)
    print(f"Null models: {null_models}")
    print(f"Statistics: {stat_names}")
    print(f"Bin configs: {bin_configs}")
    print(f"Permutations: {n_permutations}, Bootstrap: {n_bootstrap}")
    print()

    # Load data
    df = load_and_prepare_data()
    print(f"Dataset: {len(df)} samples, {df['recclass'].nunique()} classes")
    print()

    # Get classes with sufficient samples
    class_counts = df["recclass"].value_counts()
    target_classes = class_counts[class_counts >= min_samples].index.tolist()
    print(f"Classes with n≥{min_samples}: {len(target_classes)}")
    print()

    # Initialize results storage
    class_results: Dict[str, ClassResult] = {}

    # Primary bin config for main analysis
    primary_bins = bin_configs[0] if bin_configs else 10

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Compute observed statistics and bootstrap CIs
    # ═══════════════════════════════════════════════════════════════════
    print("-" * 70)
    print("PHASE 1: Computing observed statistics")
    print("-" * 70)

    for i, recclass in enumerate(target_classes):
        masses = df[df["recclass"] == recclass]["mass"].values

        # Observed stats
        stats_obs = compute_all_stats(masses)

        # Bootstrap CIs
        stats_boot = bootstrap_stats(masses, n_bootstrap, seed)

        class_results[recclass] = ClassResult(
            recclass=recclass,
            count=len(masses),
            stats_observed=stats_obs.to_dict(),
            stats_bootstrap=stats_boot,
            null_results={},
            survival={},
            total_survived=0
        )

        if (i + 1) % 50 == 0:
            print(f"  Computed stats for {i+1}/{len(target_classes)} classes")

    print(f"  Done: {len(target_classes)} classes")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Run null models
    # ═══════════════════════════════════════════════════════════════════
    print("-" * 70)
    print("PHASE 2: Running null models")
    print("-" * 70)

    all_null_results = []  # For FDR correction

    for null_name in null_models:
        print(f"\n  [{null_name.upper()}] {NULL_MODELS[null_name].description}")

        # Instantiate null model
        if null_name == "null5":
            null_model = Null5Balanced(df, n_mass_bins=primary_bins, subsample_size=subsample_size)
        else:
            null_model = get_null_model(null_name, df, n_mass_bins=primary_bins)

        for stat_name in stat_names:
            print(f"    Testing {stat_name}...", end=" ", flush=True)

            results_this = []

            for recclass in target_classes:
                result = null_model.run(
                    target_class=recclass,
                    stat_name=stat_name,
                    n_permutations=n_permutations,
                    seed=seed
                )

                # Store result
                if null_name not in class_results[recclass].null_results:
                    class_results[recclass].null_results[null_name] = {}
                class_results[recclass].null_results[null_name][stat_name] = result

                results_this.append({
                    "recclass": recclass,
                    "null_name": null_name,
                    "stat_name": stat_name,
                    "p_value": result.p_value,
                    "z_score": result.z_score,
                })

            all_null_results.extend(results_this)
            print(f"done ({len(results_this)} classes)")

    print()

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: FDR correction and survival determination
    # ═══════════════════════════════════════════════════════════════════
    print("-" * 70)
    print("PHASE 3: FDR correction and survival")
    print("-" * 70)

    # Group by (null_name, stat_name) for FDR
    results_df = pd.DataFrame(all_null_results)

    for (null_name, stat_name), group in results_df.groupby(["null_name", "stat_name"]):
        pvalues = group["p_value"].values
        q_values = apply_fdr(pvalues)

        for i, (_, row) in enumerate(group.iterrows()):
            recclass = row["recclass"]
            q_val = q_values[i]
            z_score = row["z_score"]

            # Significant if q ≤ threshold AND z < 0 (tighter than expected)
            is_significant = (q_val <= fdr_threshold) and (z_score < 0)

            survival_key = f"{null_name}_{stat_name}"
            class_results[recclass].survival[survival_key] = is_significant

    # Calculate total survived
    for recclass in target_classes:
        total = sum(1 for v in class_results[recclass].survival.values() if v)
        class_results[recclass].total_survived = total

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: Sensitivity analysis (multiple bin configs)
    # ═══════════════════════════════════════════════════════════════════
    sensitivity_results: Dict[int, Dict[str, List[str]]] = {}

    if len(bin_configs) > 1:
        print("-" * 70)
        print("PHASE 4: Sensitivity analysis")
        print("-" * 70)

        for n_bins in bin_configs:
            print(f"  Testing with {n_bins} bins...")
            sensitivity_results[n_bins] = {}

            null_model = get_null_model("null2", df, n_mass_bins=n_bins)

            for stat_name in stat_names:
                significant_classes = []

                pvalues = []
                recclasses = []

                for recclass in target_classes:
                    result = null_model.run(recclass, stat_name, n_permutations // 2, seed)
                    pvalues.append(result.p_value)
                    recclasses.append(recclass)

                q_values = apply_fdr(np.array(pvalues))

                for rc, p, q, z in zip(recclasses, pvalues,
                                       q_values,
                                       [class_results[rc].null_results.get("null2", {}).get(stat_name, NullResult("", "", 0, 0, 0, 0, 1, 0, 0, [])).z_score for rc in recclasses]):
                    if q <= fdr_threshold and z < 0:
                        significant_classes.append(rc)

                sensitivity_results[n_bins][stat_name] = significant_classes
                print(f"    {stat_name}: {len(significant_classes)} significant")

        print()

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 5: Identify robust candidates
    # ═══════════════════════════════════════════════════════════════════
    print("-" * 70)
    print("PHASE 5: Robust candidate identification")
    print("-" * 70)

    # Number of tests per class
    n_tests = len(null_models) * len(stat_names)

    # Robust = survive ALL tests
    robust_candidates = [
        rc for rc, cr in class_results.items()
        if cr.total_survived == n_tests
    ]

    # Survival summary
    survival_counts = {}
    for cr in class_results.values():
        survived = cr.total_survived
        survival_counts[survived] = survival_counts.get(survived, 0) + 1

    print(f"  Total tests per class: {n_tests}")
    print(f"  Survival distribution:")
    for k in sorted(survival_counts.keys(), reverse=True):
        print(f"    Survived {k}/{n_tests}: {survival_counts[k]} classes")

    print(f"\n  ROBUST CANDIDATES (survive ALL {n_tests} tests): {len(robust_candidates)}")
    for rc in robust_candidates:
        cr = class_results[rc]
        print(f"    ★ {rc:20} n={cr.count}")

    print()

    # ═══════════════════════════════════════════════════════════════════
    # BUILD REPORT
    # ═══════════════════════════════════════════════════════════════════
    report = BatteryReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        parameters={
            "null_spec": null_spec,
            "n_permutations": n_permutations,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "min_samples": min_samples,
            "fdr_threshold": fdr_threshold,
            "subsample_size": subsample_size,
        },
        null_models_used=null_models,
        stats_used=stat_names,
        bin_configs=bin_configs,
        total_classes_tested=len(target_classes),
        class_results=class_results,
        survival_summary=survival_counts,
        robust_candidates=robust_candidates,
        sensitivity_results=sensitivity_results,
    )

    return report


def save_battery_report(report: BatteryReport, output_dir: Path = REPORTS) -> Dict[str, str]:
    """Save battery report to multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"battery_{timestamp}"

    # 1. JSON summary
    json_path = output_dir / f"{base_name}_summary.json"
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, cls=NumpyEncoder)

    # 2. CSV with class results
    csv_data = []
    for rc, cr in report.class_results.items():
        row = {
            "recclass": rc,
            "count": cr.count,
            **{f"obs_{k}": v for k, v in cr.stats_observed.items()},
            "total_survived": cr.total_survived,
        }
        # Add z-scores for each null/stat combo
        for null_name, stat_results in cr.null_results.items():
            for stat_name, nr in stat_results.items():
                row[f"z_{null_name}_{stat_name}"] = nr.z_score
                row[f"p_{null_name}_{stat_name}"] = nr.p_value
                row[f"surv_{null_name}_{stat_name}"] = cr.survival.get(f"{null_name}_{stat_name}", False)
        csv_data.append(row)

    csv_df = pd.DataFrame(csv_data)
    csv_df = csv_df.sort_values("total_survived", ascending=False)
    csv_path = output_dir / f"{base_name}_results.csv"
    csv_df.to_csv(csv_path, index=False)

    # 3. Survival table (compact)
    survival_data = []
    for rc, cr in report.class_results.items():
        row = {"recclass": rc, "count": cr.count}
        for null_name in report.null_models_used:
            # Check if survives ALL stats for this null
            survives_null = all(
                cr.survival.get(f"{null_name}_{stat}", False)
                for stat in report.stats_used
            )
            row[null_name] = "✓" if survives_null else "✗"
        row["total"] = cr.total_survived
        survival_data.append(row)

    survival_df = pd.DataFrame(survival_data)
    survival_df = survival_df.sort_values("total", ascending=False)
    survival_path = output_dir / f"{base_name}_survival.csv"
    survival_df.to_csv(survival_path, index=False)

    # 4. Generate plots
    plot_path = output_dir / f"{base_name}_plot.png"
    generate_battery_plots(report, plot_path)

    # 5. Generate markdown report
    md_path = output_dir / f"{base_name}_report.md"
    generate_markdown_report(report, md_path)

    print("=" * 70)
    print("FILES GENERATED")
    print("=" * 70)
    print(f"  Summary JSON:   {json_path}")
    print(f"  Results CSV:    {csv_path}")
    print(f"  Survival table: {survival_path}")
    print(f"  Plot:           {plot_path}")
    print(f"  Report:         {md_path}")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "survival": str(survival_path),
        "plot": str(plot_path),
        "report": str(md_path),
    }


def generate_battery_plots(report: BatteryReport, output_path: Path):
    """Generate comprehensive battery visualization."""
    n_nulls = len(report.null_models_used)
    n_stats = len(report.stats_used)

    fig = plt.figure(figsize=(16, 10))

    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ─────────────────────────────────────────────────────────────────
    # Plot 1: Survival distribution
    # ─────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    n_tests = n_nulls * n_stats
    survival_x = list(range(n_tests + 1))
    survival_y = [report.survival_summary.get(i, 0) for i in survival_x]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, n_tests + 1))
    ax1.bar(survival_x, survival_y, color=colors)
    ax1.set_xlabel("Number of tests survived")
    ax1.set_ylabel("Number of classes")
    ax1.set_title(f"Survival Distribution (Total tests: {n_tests})")
    ax1.set_xticks(survival_x)

    # ─────────────────────────────────────────────────────────────────
    # Plot 2: Z-scores for robust candidates
    # ─────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    if report.robust_candidates:
        robust_data = []
        for rc in report.robust_candidates[:10]:  # Top 10
            cr = report.class_results[rc]
            for null_name in report.null_models_used:
                for stat_name in report.stats_used:
                    nr = cr.null_results.get(null_name, {}).get(stat_name)
                    if nr:
                        robust_data.append({
                            "class": rc,
                            "null": null_name,
                            "stat": stat_name,
                            "z": nr.z_score
                        })

        if robust_data:
            robust_df = pd.DataFrame(robust_data)
            robust_pivot = robust_df.pivot_table(
                index="class",
                columns=["null", "stat"],
                values="z"
            )

            im = ax2.imshow(robust_pivot.values, cmap="RdBu_r", aspect="auto",
                           vmin=-5, vmax=5)
            ax2.set_yticks(range(len(robust_pivot.index)))
            ax2.set_yticklabels(robust_pivot.index)
            ax2.set_xticks(range(len(robust_pivot.columns)))
            ax2.set_xticklabels([f"{n[0][:2]}-{n[1][:2]}" for n in robust_pivot.columns],
                               rotation=45, ha="right")
            plt.colorbar(im, ax=ax2, label="Z-score")
            ax2.set_title("Z-scores: Robust Candidates")
    else:
        ax2.text(0.5, 0.5, "No robust candidates", ha="center", va="center",
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title("Z-scores: Robust Candidates")

    # ─────────────────────────────────────────────────────────────────
    # Plot 3: Statistics comparison
    # ─────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    if len(report.stats_used) >= 2:
        stat1, stat2 = report.stats_used[0], report.stats_used[1]
        z1 = []
        z2 = []
        survived = []

        for rc, cr in report.class_results.items():
            nr1 = cr.null_results.get("null2", {}).get(stat1)
            nr2 = cr.null_results.get("null2", {}).get(stat2)
            if nr1 and nr2:
                z1.append(nr1.z_score)
                z2.append(nr2.z_score)
                survived.append(cr.total_survived)

        z1, z2, survived = np.array(z1), np.array(z2), np.array(survived)
        n_tests = len(report.null_models_used) * len(report.stats_used)

        scatter = ax3.scatter(z1, z2, c=survived, cmap="RdYlGn",
                             vmin=0, vmax=n_tests, alpha=0.6, s=30)
        ax3.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax3.axvline(0, color="gray", linestyle="--", alpha=0.3)
        ax3.set_xlabel(f"Z-score ({stat1})")
        ax3.set_ylabel(f"Z-score ({stat2})")
        ax3.set_title(f"Null-2: {stat1} vs {stat2}")
        plt.colorbar(scatter, ax=ax3, label="Tests survived")

    # ─────────────────────────────────────────────────────────────────
    # Plot 4: Sensitivity analysis
    # ─────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    if report.sensitivity_results:
        bin_configs = sorted(report.sensitivity_results.keys())
        stat_names = report.stats_used

        x = np.arange(len(bin_configs))
        width = 0.8 / len(stat_names)

        for i, stat in enumerate(stat_names):
            counts = [len(report.sensitivity_results[b].get(stat, [])) for b in bin_configs]
            ax4.bar(x + i * width, counts, width, label=stat)

        ax4.set_xlabel("Number of mass bins")
        ax4.set_ylabel("Significant classes")
        ax4.set_title("Sensitivity: Varying bin count")
        ax4.set_xticks(x + width * (len(stat_names) - 1) / 2)
        ax4.set_xticklabels(bin_configs)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No sensitivity analysis", ha="center", va="center",
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title("Sensitivity Analysis")

    plt.suptitle("ORIGINMAP Null Model Battery Results", fontsize=14, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_markdown_report(report: BatteryReport, output_path: Path):
    """Generate markdown report."""
    n_tests = len(report.null_models_used) * len(report.stats_used)

    lines = [
        "# ORIGINMAP Battery Report",
        "",
        f"**Generated**: {report.timestamp_utc}",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Null models | {', '.join(report.null_models_used)} |",
        f"| Statistics | {', '.join(report.stats_used)} |",
        f"| Permutations | {report.parameters['n_permutations']} |",
        f"| Bootstrap | {report.parameters['n_bootstrap']} |",
        f"| FDR threshold | {report.parameters['fdr_threshold']} |",
        f"| Min samples | {report.parameters['min_samples']} |",
        "",
        "## Null Model Descriptions",
        "",
        "| Model | Description |",
        "|-------|-------------|",
        "| null1 | Global permutation (baseline) |",
        "| null2 | Mass-bin stratified |",
        "| null3 | Mass × Time stratified |",
        "| null4 | Mass × Fall/Found stratified |",
        "| null5 | Balanced subsampling |",
        "",
        "## Statistics Descriptions",
        "",
        "| Statistic | Description |",
        "|-----------|-------------|",
        "| cv | Coefficient of Variation (std/mean) |",
        "| varlog | Variance of log(mass) |",
        "| mad_ratio | MAD / Median |",
        "",
        "## Results Summary",
        "",
        f"- **Classes tested**: {report.total_classes_tested}",
        f"- **Total tests per class**: {n_tests}",
        f"- **Robust candidates** (survive all {n_tests}): {len(report.robust_candidates)}",
        "",
        "### Survival Distribution",
        "",
        "| Survived | Classes |",
        "|----------|---------|",
    ]

    for k in sorted(report.survival_summary.keys(), reverse=True):
        lines.append(f"| {k}/{n_tests} | {report.survival_summary[k]} |")

    lines.extend([
        "",
        "## Robust Candidates",
        "",
    ])

    if report.robust_candidates:
        lines.append("| Class | n | CV | varlog | MAD/med |")
        lines.append("|-------|---|----|----|---------|")

        for rc in report.robust_candidates:
            cr = report.class_results[rc]
            cv = cr.stats_observed.get("cv", 0)
            varlog = cr.stats_observed.get("varlog", 0)
            mad = cr.stats_observed.get("mad_ratio", 0)
            lines.append(f"| {rc} | {cr.count} | {cv:.2f} | {varlog:.2f} | {mad:.2f} |")
    else:
        lines.append("*No classes survived all tests.*")

    # Sensitivity section
    if report.sensitivity_results:
        lines.extend([
            "",
            "## Sensitivity Analysis",
            "",
            "Classes significant across all bin configurations:",
            "",
        ])

        # Find intersection across all bin configs
        for stat in report.stats_used:
            all_bins_robust = None
            for n_bins, stat_results in report.sensitivity_results.items():
                classes = set(stat_results.get(stat, []))
                if all_bins_robust is None:
                    all_bins_robust = classes
                else:
                    all_bins_robust &= classes

            if all_bins_robust:
                lines.append(f"**{stat}**: {', '.join(sorted(all_bins_robust))}")
            else:
                lines.append(f"**{stat}**: None")

    lines.extend([
        "",
        "---",
        "",
        "*Report generated by ORIGINMAP hypothesis testing module*",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def run_quick_battery(n_permutations: int = 500, seed: int = 42) -> Dict[str, Any]:
    """
    Quick battery with default settings (backward compatible).
    """
    report = run_battery(
        null_spec="2-4",
        stat_names=["cv"],
        bin_configs=[10],
        n_permutations=n_permutations,
        n_bootstrap=0,
        seed=seed,
    )
    files = save_battery_report(report)
    return {
        "robust_count": len(report.robust_candidates),
        "robust_candidates": report.robust_candidates,
        **files
    }


# Backward compatible interface
def run_full_battery(n_permutations: int = 500, seed: int = 42, n_mass_bins: int = 10) -> Dict[str, Any]:
    """Backward compatible wrapper."""
    return run_quick_battery(n_permutations, seed)


if __name__ == "__main__":
    # Default: comprehensive battery
    report = run_battery(
        null_spec="1-5",
        stat_names=["cv", "varlog", "mad_ratio"],
        bin_configs=[8, 10, 12, 20],
        n_permutations=500,
        n_bootstrap=200,
        seed=42
    )
    files = save_battery_report(report)
