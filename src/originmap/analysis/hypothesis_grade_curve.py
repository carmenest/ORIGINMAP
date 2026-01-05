"""
O-Δ6: Heterogeneity Curvature vs Petrologic Grade

Formal quantitative test of H-FRAG-2 hypothesis:
"Heterogeneity is NOT monotonic with grade; Grade 5 is a maximum"

Definitions:
- G ∈ {3, 4, 5, 6} (petrologic grade)
- H1(G) = var(log(mass))
- H2(G) = MAD(log(mass)) / median(log(mass))

Null hypothesis (H0-Δ6):
  H(G) is monotonic (increasing/decreasing) or flat

Alternative (H1-Δ6):
  H(5) > H(4) AND H(5) > H(6) — interior maximum
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
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


def extract_petrologic_grade(recclass: str) -> Optional[int]:
    """
    Extract petrologic grade from recclass string.

    Examples:
        L6 -> 6
        H5 -> 5
        LL4 -> 4
        L3.2 -> 3
        H/L4 -> 4
        L~5 -> 5
        L6-melt breccia -> 6

    Returns None if grade cannot be determined.
    """
    if pd.isna(recclass):
        return None

    recclass = str(recclass).strip()

    # Pattern: look for digit 3-6 after common prefixes
    # Ordinary chondrites: L, H, LL, E, EH, EL, etc.
    patterns = [
        r'^[LHE]+[/~]?(\d)',      # L6, H5, LL4, E3, EH3, EL6
        r'^[LHE]+\d[.-](\d)',      # L3.2 -> second digit not relevant, take first
        r'(\d)[.-]\d',             # 4-5 ranges, take first
    ]

    # Direct match for grade at end of type prefix
    match = re.search(r'^[A-Za-z/]+[~]?(\d)', recclass)
    if match:
        grade = int(match.group(1))
        if 3 <= grade <= 6:
            return grade

    # Fallback: any digit 3-6 in string
    for char in recclass:
        if char.isdigit():
            grade = int(char)
            if 3 <= grade <= 6:
                return grade

    return None


def extract_chondrite_type(recclass: str) -> Optional[str]:
    """
    Extract chondrite type (L, H, LL, E, EH, EL) from recclass.
    Returns None if not an ordinary/enstatite chondrite.
    """
    if pd.isna(recclass):
        return None

    recclass = str(recclass).strip().upper()

    # Order matters: check longer patterns first
    type_patterns = ['LL', 'EH', 'EL', 'L', 'H', 'E']

    for t in type_patterns:
        if recclass.startswith(t):
            # Verify it's followed by a digit (grade)
            remainder = recclass[len(t):]
            if remainder and (remainder[0].isdigit() or remainder[0] in '/~'):
                return t

    return None


def variance_of_log(masses: np.ndarray) -> float:
    """Variance of log-transformed mass."""
    masses = masses[~np.isnan(masses)]
    masses = masses[masses > 0]
    if len(masses) < 2:
        return np.nan
    return np.var(np.log(masses), ddof=1)


def mad_log_ratio(masses: np.ndarray) -> float:
    """MAD(log(mass)) / median(log(mass))."""
    masses = masses[~np.isnan(masses)]
    masses = masses[masses > 0]
    if len(masses) < 2:
        return np.nan

    log_masses = np.log(masses)
    median = np.median(log_masses)

    if median == 0:
        return np.nan

    mad = np.median(np.abs(log_masses - median))
    return mad / abs(median)


METRICS = {
    "varlog": variance_of_log,
    "mad": mad_log_ratio,
}


@dataclass
class GradeCurveResult:
    """Results from O-Δ6 experiment."""
    metric_name: str
    grades: List[int]
    H_observed: Dict[int, float]  # grade -> H value
    n_per_grade: Dict[int, int]

    # Interior maximum test
    has_interior_max: bool  # H(5) > H(4) AND H(5) > H(6)
    delta_5_4: float        # H(5) - H(4)
    delta_5_6: float        # H(5) - H(6)

    # Permutation test
    p_value_interior_max: float
    n_permutations: int
    null_interior_max_count: int

    # Bootstrap
    bootstrap_H: Dict[int, Tuple[float, float, float]]  # grade -> (mean, ci_low, ci_high)
    bootstrap_stable: bool  # H(5) CI doesn't overlap with H(4), H(6)

    def to_dict(self) -> Dict:
        return {
            "metric_name": self.metric_name,
            "grades": self.grades,
            "H_observed": self.H_observed,
            "n_per_grade": self.n_per_grade,
            "has_interior_max": self.has_interior_max,
            "delta_5_4": self.delta_5_4,
            "delta_5_6": self.delta_5_6,
            "p_value_interior_max": self.p_value_interior_max,
            "n_permutations": self.n_permutations,
            "null_interior_max_count": self.null_interior_max_count,
            "bootstrap_H": {
                str(k): list(v) for k, v in self.bootstrap_H.items()
            },
            "bootstrap_stable": self.bootstrap_stable,
        }


def compute_H_by_grade(
    df: pd.DataFrame,
    metric_fn,
    grades: List[int] = [3, 4, 5, 6]
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    Compute heterogeneity metric by grade.

    Returns:
        H_values: dict grade -> H value
        n_values: dict grade -> sample count
    """
    H_values = {}
    n_values = {}

    for g in grades:
        masses = df[df["grade"] == g]["mass"].values
        H_values[g] = metric_fn(masses)
        n_values[g] = len(masses)

    return H_values, n_values


def test_interior_maximum(H: Dict[int, float]) -> Tuple[bool, float, float]:
    """
    Test if H(5) > H(4) AND H(5) > H(6).

    Returns:
        has_max: bool
        delta_5_4: H(5) - H(4)
        delta_5_6: H(5) - H(6)
    """
    H5 = H.get(5, np.nan)
    H4 = H.get(4, np.nan)
    H6 = H.get(6, np.nan)

    if np.isnan(H5) or np.isnan(H4) or np.isnan(H6):
        return False, np.nan, np.nan

    delta_5_4 = H5 - H4
    delta_5_6 = H5 - H6

    has_max = (H5 > H4) and (H5 > H6)

    return has_max, delta_5_4, delta_5_6


def permutation_test_interior_max(
    df: pd.DataFrame,
    metric_fn,
    observed_delta_min: float,
    n_permutations: int = 1000,
    seed: int = 42,
    grades: List[int] = [3, 4, 5, 6]
) -> Tuple[float, int]:
    """
    Permutation test for interior maximum.

    Permute grade labels, compute H(G), count how often
    we get an interior max as strong as observed.

    observed_delta_min = min(H(5)-H(4), H(5)-H(6)) observed

    Returns:
        p_value
        count of null >= observed
    """
    rng = np.random.default_rng(seed)

    # Subset to grades we care about
    df_grades = df[df["grade"].isin(grades)].copy()
    masses = df_grades["mass"].values
    original_grades = df_grades["grade"].values

    count_ge = 0

    for _ in range(n_permutations):
        # Permute grade labels
        perm_grades = rng.permutation(original_grades)

        # Compute H for each grade
        H_perm = {}
        for g in grades:
            mask = perm_grades == g
            if mask.sum() > 1:
                H_perm[g] = metric_fn(masses[mask])
            else:
                H_perm[g] = np.nan

        # Check interior max
        H5 = H_perm.get(5, np.nan)
        H4 = H_perm.get(4, np.nan)
        H6 = H_perm.get(6, np.nan)

        if not (np.isnan(H5) or np.isnan(H4) or np.isnan(H6)):
            perm_delta_min = min(H5 - H4, H5 - H6)
            if perm_delta_min >= observed_delta_min:
                count_ge += 1

    p_value = count_ge / n_permutations
    return p_value, count_ge


def bootstrap_H_by_grade(
    df: pd.DataFrame,
    metric_fn,
    n_bootstrap: int = 500,
    seed: int = 42,
    grades: List[int] = [3, 4, 5, 6],
    confidence: float = 0.95
) -> Dict[int, Tuple[float, float, float]]:
    """
    Bootstrap confidence intervals for H(G).

    Returns:
        dict grade -> (mean, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2

    results = {}

    for g in grades:
        masses = df[df["grade"] == g]["mass"].values
        n = len(masses)

        if n < 10:
            results[g] = (np.nan, np.nan, np.nan)
            continue

        boot_H = []
        for _ in range(n_bootstrap):
            sample = rng.choice(masses, size=n, replace=True)
            boot_H.append(metric_fn(sample))

        boot_H = np.array(boot_H)
        boot_H = boot_H[~np.isnan(boot_H)]

        if len(boot_H) < 10:
            results[g] = (np.nan, np.nan, np.nan)
            continue

        results[g] = (
            np.mean(boot_H),
            np.percentile(boot_H, alpha * 100),
            np.percentile(boot_H, (1 - alpha) * 100)
        )

    return results


def check_bootstrap_stable(
    bootstrap_H: Dict[int, Tuple[float, float, float]]
) -> bool:
    """
    Check if Grade 5 CI doesn't overlap with Grade 4 and 6 CIs.

    Stable = H5_ci_low > max(H4_ci_high, H6_ci_high)
    """
    H5 = bootstrap_H.get(5, (np.nan, np.nan, np.nan))
    H4 = bootstrap_H.get(4, (np.nan, np.nan, np.nan))
    H6 = bootstrap_H.get(6, (np.nan, np.nan, np.nan))

    if any(np.isnan(H5)) or any(np.isnan(H4)) or any(np.isnan(H6)):
        return False

    # H5 lower bound > H4 upper bound AND H5 lower bound > H6 upper bound
    return (H5[1] > H4[2]) and (H5[1] > H6[2])


def balanced_subsample_test(
    df: pd.DataFrame,
    metric_fn,
    n_samples: int = 500,
    n_bootstrap: int = 100,
    seed: int = 42,
    grades: List[int] = [3, 4, 5, 6]
) -> Dict[int, Tuple[float, float, float]]:
    """
    Test with balanced subsampling per grade.

    Subsample each grade to same n, repeat K times.
    Returns distribution of H(G).
    """
    rng = np.random.default_rng(seed)

    # Find minimum n across grades
    n_per_grade = {}
    for g in grades:
        n_per_grade[g] = (df["grade"] == g).sum()

    min_n = min(n_per_grade.values())
    subsample_n = min(n_samples, min_n)

    if subsample_n < 10:
        return {g: (np.nan, np.nan, np.nan) for g in grades}

    results = {g: [] for g in grades}

    for _ in range(n_bootstrap):
        for g in grades:
            masses = df[df["grade"] == g]["mass"].values
            sample = rng.choice(masses, size=subsample_n, replace=False)
            results[g].append(metric_fn(sample))

    final = {}
    for g in grades:
        vals = np.array(results[g])
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 10:
            final[g] = (np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5))
        else:
            final[g] = (np.nan, np.nan, np.nan)

    return final


def run_grade_curve_experiment(
    metric_name: str = "varlog",
    n_permutations: int = 1000,
    n_bootstrap: int = 500,
    seed: int = 42,
    chondrite_types: Optional[List[str]] = None,
    min_per_grade: int = 30,
) -> GradeCurveResult:
    """
    Run O-Δ6 experiment for a single metric.

    Args:
        metric_name: 'varlog' or 'mad'
        n_permutations: number of permutations for null test
        n_bootstrap: number of bootstrap samples
        seed: random seed
        chondrite_types: if provided, filter to these types (e.g., ['L', 'H', 'LL'])
        min_per_grade: minimum samples per grade

    Returns:
        GradeCurveResult
    """
    # Load data
    df = pd.read_parquet(DATA_PROCESSED / "meteorites.parquet")
    df = df.dropna(subset=["mass", "recclass"])
    df = df[df["mass"] > 0].copy()

    # Extract grade
    df["grade"] = df["recclass"].apply(extract_petrologic_grade)
    df["chondrite_type"] = df["recclass"].apply(extract_chondrite_type)

    # Filter to valid grades
    df = df[df["grade"].isin([3, 4, 5, 6])].copy()

    # Filter to chondrite types if specified
    if chondrite_types:
        df = df[df["chondrite_type"].isin(chondrite_types)].copy()

    # Check minimum samples
    grade_counts = df["grade"].value_counts()
    valid_grades = [g for g in [3, 4, 5, 6] if grade_counts.get(g, 0) >= min_per_grade]

    if len(valid_grades) < 3 or 5 not in valid_grades:
        raise ValueError(f"Insufficient data. Grade counts: {grade_counts.to_dict()}")

    grades = sorted(valid_grades)
    metric_fn = METRICS[metric_name]

    # Compute observed H
    H_observed, n_per_grade = compute_H_by_grade(df, metric_fn, grades)

    # Test interior maximum
    has_interior_max, delta_5_4, delta_5_6 = test_interior_maximum(H_observed)
    observed_delta_min = min(delta_5_4, delta_5_6) if has_interior_max else -float('inf')

    # Permutation test
    p_value, null_count = permutation_test_interior_max(
        df, metric_fn, observed_delta_min, n_permutations, seed, grades
    )

    # Bootstrap
    bootstrap_H = bootstrap_H_by_grade(df, metric_fn, n_bootstrap, seed, grades)
    bootstrap_stable = check_bootstrap_stable(bootstrap_H)

    return GradeCurveResult(
        metric_name=metric_name,
        grades=grades,
        H_observed=H_observed,
        n_per_grade=n_per_grade,
        has_interior_max=has_interior_max,
        delta_5_4=delta_5_4,
        delta_5_6=delta_5_6,
        p_value_interior_max=p_value,
        n_permutations=n_permutations,
        null_interior_max_count=null_count,
        bootstrap_H=bootstrap_H,
        bootstrap_stable=bootstrap_stable,
    )


def run_full_experiment(
    n_permutations: int = 1000,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run O-Δ6 for both metrics, global and by type.
    """
    print("=" * 70)
    print("O-Δ6: Heterogeneity Curvature vs Petrologic Grade")
    print("=" * 70)
    print(f"Permutations: {n_permutations}, Bootstrap: {n_bootstrap}, Seed: {seed}")
    print()

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "n_permutations": n_permutations,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        },
        "metrics": {},
        "by_type": {},
    }

    # Global analysis
    print("-" * 70)
    print("GLOBAL ANALYSIS (all ordinary chondrites)")
    print("-" * 70)

    for metric_name in ["varlog", "mad"]:
        print(f"\n  [{metric_name.upper()}]")

        try:
            result = run_grade_curve_experiment(
                metric_name=metric_name,
                n_permutations=n_permutations,
                n_bootstrap=n_bootstrap,
                seed=seed,
                chondrite_types=["L", "H", "LL"],
            )

            results["metrics"][metric_name] = result.to_dict()

            print(f"    n per grade: {result.n_per_grade}")
            print(f"    H observed: {', '.join(f'G{g}={result.H_observed[g]:.4f}' for g in result.grades)}")
            print(f"    Interior max (H5>H4,H6): {result.has_interior_max}")
            print(f"    Δ(5-4)={result.delta_5_4:.4f}, Δ(5-6)={result.delta_5_6:.4f}")
            print(f"    p-value: {result.p_value_interior_max:.4f}")
            print(f"    Bootstrap stable: {result.bootstrap_stable}")

        except Exception as e:
            print(f"    Error: {e}")
            results["metrics"][metric_name] = {"error": str(e)}

    # By type analysis
    print()
    print("-" * 70)
    print("BY TYPE ANALYSIS")
    print("-" * 70)

    for chondrite_type in ["L", "H", "LL"]:
        results["by_type"][chondrite_type] = {}
        print(f"\n  TYPE: {chondrite_type}")

        for metric_name in ["varlog", "mad"]:
            try:
                result = run_grade_curve_experiment(
                    metric_name=metric_name,
                    n_permutations=n_permutations // 2,  # Faster
                    n_bootstrap=n_bootstrap // 2,
                    seed=seed,
                    chondrite_types=[chondrite_type],
                    min_per_grade=20,
                )

                results["by_type"][chondrite_type][metric_name] = result.to_dict()

                status = "✓" if result.has_interior_max and result.p_value_interior_max <= 0.05 else "✗"
                print(f"    {metric_name}: {status} interior_max={result.has_interior_max}, p={result.p_value_interior_max:.3f}")

            except Exception as e:
                print(f"    {metric_name}: insufficient data")
                results["by_type"][chondrite_type][metric_name] = {"error": str(e)}

    return results


def generate_outputs(results: Dict[str, Any], output_dir: Path = REPORTS) -> Dict[str, str]:
    """Generate all O-Δ6 outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # 1. CSV: Heterogeneity by grade
    csv_data = []
    for metric_name, metric_result in results["metrics"].items():
        if "error" in metric_result:
            continue
        for g in metric_result["grades"]:
            csv_data.append({
                "metric": metric_name,
                "grade": g,
                "H_observed": metric_result["H_observed"].get(str(g), np.nan),
                "n": metric_result["n_per_grade"].get(str(g), 0),
                "bootstrap_mean": metric_result["bootstrap_H"].get(str(g), [np.nan])[0],
                "bootstrap_ci_low": metric_result["bootstrap_H"].get(str(g), [np.nan, np.nan])[1],
                "bootstrap_ci_high": metric_result["bootstrap_H"].get(str(g), [np.nan, np.nan, np.nan])[2],
            })

    csv_df = pd.DataFrame(csv_data)
    csv_path = output_dir / "O-D6_heterogeneity_by_grade.csv"
    csv_df.to_csv(csv_path, index=False)
    files["csv"] = str(csv_path)

    # 2. JSON: P-values and test results
    pvalues = {
        "experiment": "O-D6",
        "hypothesis": "H-FRAG-2: Interior maximum at Grade 5",
        "timestamp_utc": results["timestamp_utc"],
        "parameters": results["parameters"],
        "global_results": {},
        "by_type_results": {},
    }

    for metric_name, metric_result in results["metrics"].items():
        if "error" not in metric_result:
            pvalues["global_results"][metric_name] = {
                "has_interior_max": metric_result["has_interior_max"],
                "p_value": metric_result["p_value_interior_max"],
                "bootstrap_stable": metric_result["bootstrap_stable"],
                "delta_5_4": metric_result["delta_5_4"],
                "delta_5_6": metric_result["delta_5_6"],
            }

    for chondrite_type, type_results in results["by_type"].items():
        pvalues["by_type_results"][chondrite_type] = {}
        for metric_name, metric_result in type_results.items():
            if "error" not in metric_result:
                pvalues["by_type_results"][chondrite_type][metric_name] = {
                    "has_interior_max": metric_result["has_interior_max"],
                    "p_value": metric_result["p_value_interior_max"],
                }

    json_path = output_dir / "O-D6_perm_pvalues.json"
    with open(json_path, "w") as f:
        json.dump(pvalues, f, indent=2, cls=NumpyEncoder)
    files["json"] = str(json_path)

    # 3. Bootstrap CSV
    boot_data = []
    for metric_name, metric_result in results["metrics"].items():
        if "error" in metric_result:
            continue
        for g, (mean, ci_low, ci_high) in metric_result["bootstrap_H"].items():
            boot_data.append({
                "metric": metric_name,
                "grade": g,
                "bootstrap_mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

    boot_df = pd.DataFrame(boot_data)
    boot_path = output_dir / "O-D6_bootstrap.csv"
    boot_df.to_csv(boot_path, index=False)
    files["bootstrap"] = str(boot_path)

    # 4. Plot
    plot_path = output_dir / "O-D6_curve.png"
    generate_curve_plot(results, plot_path)
    files["plot"] = str(plot_path)

    print()
    print("=" * 70)
    print("FILES GENERATED")
    print("=" * 70)
    for k, v in files.items():
        print(f"  {k}: {v}")

    return files


def generate_curve_plot(results: Dict[str, Any], output_path: Path):
    """Generate curve plot with bootstrap bands."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    grades = [3, 4, 5, 6]
    colors = {"varlog": "steelblue", "mad": "darkorange"}

    for ax_idx, metric_name in enumerate(["varlog", "mad"]):
        ax = axes[ax_idx]
        metric_result = results["metrics"].get(metric_name, {})

        if "error" in metric_result:
            ax.text(0.5, 0.5, f"No data for {metric_name}", ha="center", va="center",
                   transform=ax.transAxes)
            continue

        # Extract data
        H_obs = [metric_result["H_observed"].get(str(g), np.nan) for g in grades]
        boot = metric_result["bootstrap_H"]

        means = [boot.get(str(g), [np.nan, np.nan, np.nan])[0] for g in grades]
        ci_low = [boot.get(str(g), [np.nan, np.nan, np.nan])[1] for g in grades]
        ci_high = [boot.get(str(g), [np.nan, np.nan, np.nan])[2] for g in grades]

        # Plot bootstrap band
        ax.fill_between(grades, ci_low, ci_high, alpha=0.3, color=colors[metric_name],
                       label="95% CI")

        # Plot observed line
        ax.plot(grades, H_obs, 'o-', color=colors[metric_name], linewidth=2,
               markersize=10, label="Observed")

        # Mark grade 5
        if 5 in grades:
            idx = grades.index(5)
            ax.axvline(x=5, color="red", linestyle="--", alpha=0.5)
            ax.scatter([5], [H_obs[idx]], s=200, marker="*", color="red",
                      zorder=5, label="Grade 5 (test)")

        # Labels
        ax.set_xlabel("Petrologic Grade", fontsize=12)
        ax.set_ylabel(f"H ({metric_name})", fontsize=12)
        ax.set_xticks(grades)
        ax.set_xticklabels([f"G{g}" for g in grades])

        # Title with test result
        has_max = metric_result.get("has_interior_max", False)
        p_val = metric_result.get("p_value_interior_max", 1.0)
        stable = metric_result.get("bootstrap_stable", False)

        status = "SUPPORTED" if (has_max and p_val <= 0.05 and stable) else \
                 "PARTIAL" if has_max else "REJECTED"

        ax.set_title(f"{metric_name.upper()}: Interior Max = {has_max}\n"
                    f"p = {p_val:.4f}, stable = {stable}\n"
                    f"[{status}]", fontsize=11)

        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("O-Δ6: Heterogeneity Curvature vs Petrologic Grade\n"
                "H1-Δ6: H(5) > H(4) AND H(5) > H(6)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_observation_md(results: Dict[str, Any], files: Dict[str, str], output_dir: Path) -> str:
    """Generate observation markdown."""
    date = datetime.now().strftime("%Y%m%d")
    md_path = output_dir / f"observation_O-D6_{date}.md"

    lines = [
        "# Observation O-Δ6: Heterogeneity Curvature vs Grade",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Experiment**: O-Δ6 (Grade Curve)",
        f"**Permutations**: {results['parameters']['n_permutations']}",
        f"**Bootstrap**: {results['parameters']['n_bootstrap']}",
        f"**Seed**: {results['parameters']['seed']}",
        "",
        "---",
        "",
        "## Hypothesis",
        "",
        "**H0-Δ6**: H(G) is monotonic or flat",
        "",
        "**H1-Δ6**: H(5) > H(4) AND H(5) > H(6) (interior maximum)",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "### Global (L + H + LL)",
        "",
        "| Metric | Interior Max | Δ(5-4) | Δ(5-6) | p-value | Bootstrap Stable |",
        "|--------|-------------|--------|--------|---------|------------------|",
    ]

    for metric_name in ["varlog", "mad"]:
        r = results["metrics"].get(metric_name, {})
        if "error" not in r:
            lines.append(
                f"| {metric_name} | {r['has_interior_max']} | {r['delta_5_4']:.4f} | "
                f"{r['delta_5_6']:.4f} | {r['p_value_interior_max']:.4f} | {r['bootstrap_stable']} |"
            )

    lines.extend([
        "",
        "### By Chondrite Type",
        "",
        "| Type | Metric | Interior Max | p-value |",
        "|------|--------|-------------|---------|",
    ])

    for ctype in ["L", "H", "LL"]:
        for metric_name in ["varlog", "mad"]:
            r = results["by_type"].get(ctype, {}).get(metric_name, {})
            if "error" not in r:
                lines.append(
                    f"| {ctype} | {metric_name} | {r['has_interior_max']} | {r['p_value_interior_max']:.3f} |"
                )
            else:
                lines.append(f"| {ctype} | {metric_name} | - | - |")

    # Verdict
    varlog_r = results["metrics"].get("varlog", {})
    mad_r = results["metrics"].get("mad", {})

    varlog_pass = varlog_r.get("has_interior_max", False) and varlog_r.get("p_value_interior_max", 1) <= 0.05
    mad_pass = mad_r.get("has_interior_max", False) and mad_r.get("p_value_interior_max", 1) <= 0.05

    if varlog_pass and mad_pass:
        verdict = "STRONGLY SUPPORTED"
    elif varlog_pass or mad_pass:
        verdict = "PARTIALLY SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"

    lines.extend([
        "",
        "---",
        "",
        "## Verdict",
        "",
        f"**H1-Δ6 (Interior maximum at Grade 5)**: {verdict}",
        "",
        "| Condition | Status |",
        "|-----------|--------|",
        f"| varlog: H(5) > H(4), H(6) with p ≤ 0.05 | {'✓' if varlog_pass else '✗'} |",
        f"| mad: H(5) > H(4), H(6) with p ≤ 0.05 | {'✓' if mad_pass else '✗'} |",
        f"| Bootstrap stability | {'✓' if varlog_r.get('bootstrap_stable', False) else '✗'} |",
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
        "*Generated by ORIGINMAP O-Δ6 experiment*",
    ])

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return str(md_path)


def run_o_delta_6(
    n_permutations: int = 1000,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Main entry point for O-Δ6 experiment.
    """
    results = run_full_experiment(n_permutations, n_bootstrap, seed)
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
    print("O-Δ6 VERDICT")
    print("=" * 70)

    varlog_r = results["metrics"].get("varlog", {})
    mad_r = results["metrics"].get("mad", {})

    varlog_pass = varlog_r.get("has_interior_max", False) and varlog_r.get("p_value_interior_max", 1) <= 0.05
    mad_pass = mad_r.get("has_interior_max", False) and mad_r.get("p_value_interior_max", 1) <= 0.05
    boot_stable = varlog_r.get("bootstrap_stable", False)

    print(f"  varlog interior max: {'✓' if varlog_pass else '✗'} (p={varlog_r.get('p_value_interior_max', np.nan):.4f})")
    print(f"  mad interior max:    {'✓' if mad_pass else '✗'} (p={mad_r.get('p_value_interior_max', np.nan):.4f})")
    print(f"  Bootstrap stable:    {'✓' if boot_stable else '✗'}")
    print()

    if varlog_pass and mad_pass and boot_stable:
        print("  → H-FRAG-2 STRONGLY SUPPORTED")
        print("    Grade 5 is a robust interior maximum for heterogeneity")
    elif varlog_pass or mad_pass:
        print("  → H-FRAG-2 PARTIALLY SUPPORTED")
        print("    Interior maximum observed but not consistent across metrics")
    else:
        print("  → H-FRAG-2 NOT SUPPORTED")
        print("    No significant interior maximum at Grade 5")

    return files


if __name__ == "__main__":
    run_o_delta_6(n_permutations=1000, n_bootstrap=500, seed=42)
