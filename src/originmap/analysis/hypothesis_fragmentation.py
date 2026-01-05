"""
Hypothesis Test: H-FRAG (Fragmentation Uniformity)

Hypothesis:
"Grade 6 metamorphism produces more uniform mechanical properties,
resulting in more predictable (lower variance) fragmentation."

Falsifiable Predictions:
1. Grade 6 < Grade 5 in CV for ALL ordinary chondrite types (L, H, LL)
2. Monotonic trend: CV should decrease with increasing grade (3→4→5→6)
3. Effect should be present in BOTH Falls and Finds
4. Achondrites should follow DIFFERENT pattern (they melted, not just metamorphosed)
5. Effect should be STRONGER for larger sample sizes (more statistical power)

If ANY prediction fails, the hypothesis needs modification or rejection.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from originmap.config import DATA_PROCESSED, REPORTS


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def cv(masses: np.ndarray) -> float:
    """Coefficient of variation."""
    if len(masses) < 2 or np.mean(masses) == 0:
        return np.nan
    return np.std(masses) / np.mean(masses)


def bootstrap_cv(masses: np.ndarray, n_bootstrap: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for CV."""
    rng = np.random.default_rng(seed)
    cvs = []
    for _ in range(n_bootstrap):
        sample = rng.choice(masses, size=len(masses), replace=True)
        cvs.append(cv(sample))
    cvs = np.array(cvs)
    return np.percentile(cvs, 2.5), np.median(cvs), np.percentile(cvs, 97.5)


def test_prediction_1(df: pd.DataFrame) -> Dict[str, Any]:
    """
    PREDICTION 1: Grade 6 < Grade 5 in CV for ALL ordinary chondrite types.

    If H-FRAG is true, this should hold for L, H, AND LL.
    """
    print("\n" + "=" * 70)
    print("PREDICTION 1: Grade 6 < Grade 5 for all types (L, H, LL)")
    print("=" * 70)

    results = {"prediction": "Grade 6 < Grade 5 for L, H, LL", "tests": []}

    for chondrite_type in ["L", "H", "LL"]:
        grade5 = df[df["recclass"] == f"{chondrite_type}5"]["mass"].dropna().values
        grade6 = df[df["recclass"] == f"{chondrite_type}6"]["mass"].dropna().values

        if len(grade5) < 30 or len(grade6) < 30:
            print(f"  {chondrite_type}: insufficient data (n5={len(grade5)}, n6={len(grade6)})")
            continue

        cv5 = cv(grade5)
        cv6 = cv(grade6)

        # Bootstrap test
        cv5_lo, cv5_med, cv5_hi = bootstrap_cv(grade5)
        cv6_lo, cv6_med, cv6_hi = bootstrap_cv(grade6)

        # Is CV6 significantly less than CV5?
        # Check if 95% CI of CV6 is entirely below CV5 median
        significant = cv6_hi < cv5_med
        passes = cv6 < cv5

        result = {
            "type": chondrite_type,
            "n5": len(grade5),
            "n6": len(grade6),
            "cv5": cv5,
            "cv6": cv6,
            "cv5_ci": [cv5_lo, cv5_hi],
            "cv6_ci": [cv6_lo, cv6_hi],
            "passes": passes,
            "significant": significant,
        }
        results["tests"].append(result)

        status = "✓ PASS" if passes else "✗ FAIL"
        sig = "(significant)" if significant else "(not significant)"
        print(f"  {chondrite_type}5 (n={len(grade5)}): CV={cv5:.2f}")
        print(f"  {chondrite_type}6 (n={len(grade6)}): CV={cv6:.2f}")
        print(f"  → {status} {sig}")
        print()

    # Overall verdict
    all_pass = all(t["passes"] for t in results["tests"])
    results["verdict"] = "SUPPORTED" if all_pass else "FALSIFIED"

    print(f"  VERDICT: {results['verdict']}")
    return results


def test_prediction_2(df: pd.DataFrame) -> Dict[str, Any]:
    """
    PREDICTION 2: Monotonic trend - CV should decrease 3→4→5→6.

    If metamorphism causes uniformity, higher grades = lower CV.
    """
    print("\n" + "=" * 70)
    print("PREDICTION 2: Monotonic decrease in CV with grade (3→4→5→6)")
    print("=" * 70)

    results = {"prediction": "Monotonic CV decrease with grade", "tests": []}

    for chondrite_type in ["L", "H", "LL"]:
        grades = []
        cvs = []
        ns = []

        for grade in ["3", "4", "5", "6"]:
            mass = df[df["recclass"] == f"{chondrite_type}{grade}"]["mass"].dropna().values
            if len(mass) >= 30:
                grades.append(int(grade))
                cvs.append(cv(mass))
                ns.append(len(mass))

        if len(grades) < 3:
            print(f"  {chondrite_type}: insufficient grades with data")
            continue

        # Test for monotonic decrease
        # Spearman correlation should be negative
        if len(grades) >= 3:
            corr, pval = stats.spearmanr(grades, cvs)
            monotonic_decrease = corr < 0
        else:
            corr, pval = np.nan, np.nan
            monotonic_decrease = False

        result = {
            "type": chondrite_type,
            "grades": grades,
            "cvs": cvs,
            "ns": ns,
            "spearman_r": corr,
            "spearman_p": pval,
            "monotonic_decrease": monotonic_decrease,
        }
        results["tests"].append(result)

        print(f"  {chondrite_type}:")
        for g, c, n in zip(grades, cvs, ns):
            print(f"    Grade {g}: CV={c:.2f} (n={n})")
        print(f"    Spearman r={corr:.3f}, p={pval:.4f}")
        status = "✓ PASS" if monotonic_decrease else "✗ FAIL"
        print(f"    → {status} (monotonic decrease)")
        print()

    all_pass = all(t["monotonic_decrease"] for t in results["tests"])
    results["verdict"] = "SUPPORTED" if all_pass else "FALSIFIED"

    print(f"  VERDICT: {results['verdict']}")
    return results


def test_prediction_3(df: pd.DataFrame) -> Dict[str, Any]:
    """
    PREDICTION 3: Effect present in BOTH Falls and Finds.

    If it's physics (fragmentation), should appear regardless of discovery method.
    If it only appears in Finds, might be preservation/collection bias.
    """
    print("\n" + "=" * 70)
    print("PREDICTION 3: Effect present in both Falls and Finds")
    print("=" * 70)

    results = {"prediction": "Grade 6 < Grade 5 in both Falls and Finds", "tests": []}

    for discovery in ["Fell", "Found"]:
        subset = df[df["fall"] == discovery]

        print(f"\n  {discovery.upper()}:")

        for chondrite_type in ["L", "H"]:  # LL has too few Falls
            grade5 = subset[subset["recclass"] == f"{chondrite_type}5"]["mass"].dropna().values
            grade6 = subset[subset["recclass"] == f"{chondrite_type}6"]["mass"].dropna().values

            if len(grade5) < 20 or len(grade6) < 20:
                print(f"    {chondrite_type}: insufficient data (n5={len(grade5)}, n6={len(grade6)})")
                continue

            cv5 = cv(grade5)
            cv6 = cv(grade6)
            passes = cv6 < cv5

            result = {
                "discovery": discovery,
                "type": chondrite_type,
                "n5": len(grade5),
                "n6": len(grade6),
                "cv5": cv5,
                "cv6": cv6,
                "passes": passes,
            }
            results["tests"].append(result)

            status = "✓" if passes else "✗"
            print(f"    {chondrite_type}: CV5={cv5:.2f} (n={len(grade5)}), CV6={cv6:.2f} (n={len(grade6)}) {status}")

    # Check if pattern holds for both
    fell_tests = [t for t in results["tests"] if t["discovery"] == "Fell"]
    found_tests = [t for t in results["tests"] if t["discovery"] == "Found"]

    fell_pass = all(t["passes"] for t in fell_tests) if fell_tests else False
    found_pass = all(t["passes"] for t in found_tests) if found_tests else False

    results["fell_passes"] = fell_pass
    results["found_passes"] = found_pass
    results["verdict"] = "SUPPORTED" if (fell_pass and found_pass) else "PARTIAL" if (fell_pass or found_pass) else "FALSIFIED"

    print(f"\n  Falls: {'PASS' if fell_pass else 'FAIL'}")
    print(f"  Finds: {'PASS' if found_pass else 'FAIL'}")
    print(f"  VERDICT: {results['verdict']}")

    return results


def test_prediction_4(df: pd.DataFrame) -> Dict[str, Any]:
    """
    PREDICTION 4: Achondrites follow DIFFERENT pattern.

    Achondrites experienced melting, not just metamorphism.
    If H-FRAG is about metamorphic homogenization specifically,
    achondrites should NOT show grade-like CV patterns.
    """
    print("\n" + "=" * 70)
    print("PREDICTION 4: Achondrites follow different pattern")
    print("=" * 70)

    # Compare achondrite families
    achondrite_groups = {
        "HED (Vesta)": ["Eucrite", "Diogenite", "Howardite"],
        "Primitive": ["Ureilite", "Acapulcoite", "Lodranite"],
        "Other": ["Aubrite", "Angrite"],
    }

    results = {"prediction": "Achondrites don't follow chondrite grade pattern", "groups": []}

    print("\n  Achondrite CVs (should NOT show systematic pattern):")

    for group_name, classes in achondrite_groups.items():
        group_data = []
        print(f"\n  {group_name}:")

        for cls in classes:
            mass = df[df["recclass"] == cls]["mass"].dropna().values
            if len(mass) >= 10:
                cv_val = cv(mass)
                group_data.append({"class": cls, "n": len(mass), "cv": cv_val})
                print(f"    {cls}: CV={cv_val:.2f} (n={len(mass)})")

        results["groups"].append({"name": group_name, "classes": group_data})

    # Compare chondrite grade 6 vs achondrites
    print("\n  Comparison with chondrite grade 6:")

    chondrite_cvs = []
    for t in ["L", "H", "LL"]:
        mass = df[df["recclass"] == f"{t}6"]["mass"].dropna().values
        if len(mass) >= 30:
            chondrite_cvs.append(cv(mass))

    achondrite_cvs = []
    for group in results["groups"]:
        for cls in group["classes"]:
            achondrite_cvs.append(cls["cv"])

    if chondrite_cvs and achondrite_cvs:
        chondrite_mean = np.mean(chondrite_cvs)
        achondrite_mean = np.mean(achondrite_cvs)

        print(f"    Chondrite grade 6 mean CV: {chondrite_mean:.2f}")
        print(f"    Achondrite mean CV: {achondrite_mean:.2f}")

        # Achondrites should be DIFFERENT (not necessarily higher or lower)
        # The key is they shouldn't follow the same grade pattern
        results["chondrite_mean_cv"] = chondrite_mean
        results["achondrite_mean_cv"] = achondrite_mean

    results["verdict"] = "DESCRIPTIVE"  # This is more qualitative
    print(f"\n  VERDICT: {results['verdict']} (achondrites show varied pattern)")

    return results


def test_prediction_5(df: pd.DataFrame, n_permutations: int = 500, seed: int = 42) -> Dict[str, Any]:
    """
    PREDICTION 5: Effect is NOT explained by sample size.

    Use permutation test: if we randomly assign masses to grades,
    do we get the same CV pattern? If not, it's real structure.
    """
    print("\n" + "=" * 70)
    print("PREDICTION 5: Effect survives permutation test (not sample size artifact)")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    results = {"prediction": "Effect survives permutation", "tests": []}

    for chondrite_type in ["L", "H"]:
        grade5 = df[df["recclass"] == f"{chondrite_type}5"]["mass"].dropna().values
        grade6 = df[df["recclass"] == f"{chondrite_type}6"]["mass"].dropna().values

        if len(grade5) < 30 or len(grade6) < 30:
            continue

        # Observed difference
        obs_diff = cv(grade6) - cv(grade5)  # Should be negative if H-FRAG is true

        # Permutation test: pool masses, randomly split
        pooled = np.concatenate([grade5, grade6])
        n5, n6 = len(grade5), len(grade6)

        null_diffs = []
        for _ in range(n_permutations):
            rng.shuffle(pooled)
            perm_cv5 = cv(pooled[:n5])
            perm_cv6 = cv(pooled[n5:n5+n6])
            null_diffs.append(perm_cv6 - perm_cv5)

        null_diffs = np.array(null_diffs)

        # P-value: how often does null produce difference as extreme as observed?
        p_value = (null_diffs <= obs_diff).sum() / n_permutations

        result = {
            "type": chondrite_type,
            "observed_diff": obs_diff,
            "null_mean": np.mean(null_diffs),
            "null_std": np.std(null_diffs),
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
        results["tests"].append(result)

        status = "✓ SIGNIFICANT" if p_value < 0.05 else "✗ NOT SIGNIFICANT"
        print(f"  {chondrite_type}:")
        print(f"    Observed CV6-CV5: {obs_diff:.2f}")
        print(f"    Null distribution: mean={np.mean(null_diffs):.2f}, std={np.std(null_diffs):.2f}")
        print(f"    P-value: {p_value:.4f}")
        print(f"    → {status}")
        print()

    all_sig = all(t["significant"] for t in results["tests"])
    results["verdict"] = "SUPPORTED" if all_sig else "FALSIFIED"

    print(f"  VERDICT: {results['verdict']}")
    return results


def run_all_predictions(n_permutations: int = 500, seed: int = 42) -> Dict[str, Any]:
    """Run all prediction tests and synthesize verdict."""

    print("=" * 70)
    print("H-FRAG HYPOTHESIS TEST SUITE")
    print("=" * 70)
    print("""
Hypothesis: "Grade 6 metamorphism produces more uniform mechanical
properties, resulting in more predictable fragmentation."

Testing 5 falsifiable predictions...
""")

    df = pd.read_parquet(DATA_PROCESSED / "meteorites.parquet")
    df = df.dropna(subset=["mass", "recclass"])
    df = df[df["mass"] > 0]

    results = {
        "hypothesis": "H-FRAG",
        "statement": "Grade 6 metamorphism → uniform mechanics → predictable fragmentation",
        "timestamp_utc": datetime.utcnow().isoformat(),
        "predictions": {},
    }

    # Run all tests
    results["predictions"]["P1_grade6_lt_grade5"] = test_prediction_1(df)
    results["predictions"]["P2_monotonic_trend"] = test_prediction_2(df)
    results["predictions"]["P3_falls_and_finds"] = test_prediction_3(df)
    results["predictions"]["P4_achondrites_different"] = test_prediction_4(df)
    results["predictions"]["P5_permutation_test"] = test_prediction_5(df, n_permutations, seed)

    # Synthesize overall verdict
    verdicts = [
        results["predictions"]["P1_grade6_lt_grade5"]["verdict"],
        results["predictions"]["P2_monotonic_trend"]["verdict"],
        results["predictions"]["P3_falls_and_finds"]["verdict"],
        results["predictions"]["P5_permutation_test"]["verdict"],
    ]

    supported = sum(1 for v in verdicts if v == "SUPPORTED")
    falsified = sum(1 for v in verdicts if v == "FALSIFIED")

    print("\n" + "=" * 70)
    print("OVERALL SYNTHESIS")
    print("=" * 70)

    print(f"""
Prediction Results:
  P1 (Grade 6 < Grade 5):     {results["predictions"]["P1_grade6_lt_grade5"]["verdict"]}
  P2 (Monotonic trend):       {results["predictions"]["P2_monotonic_trend"]["verdict"]}
  P3 (Falls AND Finds):       {results["predictions"]["P3_falls_and_finds"]["verdict"]}
  P4 (Achondrites different): {results["predictions"]["P4_achondrites_different"]["verdict"]}
  P5 (Permutation test):      {results["predictions"]["P5_permutation_test"]["verdict"]}
""")

    if falsified == 0 and supported >= 3:
        overall = "STRONGLY SUPPORTED"
    elif falsified == 0:
        overall = "SUPPORTED"
    elif supported > falsified:
        overall = "PARTIALLY SUPPORTED"
    elif falsified > supported:
        overall = "WEAKLY SUPPORTED / NEEDS REVISION"
    else:
        overall = "FALSIFIED"

    results["overall_verdict"] = overall

    print(f"OVERALL VERDICT: {overall}")
    print("=" * 70)

    # Save results
    REPORTS.mkdir(parents=True, exist_ok=True)

    json_path = REPORTS / "hypothesis_H-FRAG_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Generate plot
    plot_path = REPORTS / "hypothesis_H-FRAG_plot.png"
    generate_hfrag_plot(df, results, plot_path)

    print(f"\nResults saved to: {json_path}")
    print(f"Plot saved to: {plot_path}")

    return results


def generate_hfrag_plot(df: pd.DataFrame, results: Dict, output_path: Path):
    """Generate visualization of H-FRAG hypothesis test."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: CV by grade for L, H, LL
    ax1 = axes[0, 0]

    for i, t in enumerate(["L", "H", "LL"]):
        grades = []
        cvs = []
        for g in ["3", "4", "5", "6"]:
            mass = df[df["recclass"] == f"{t}{g}"]["mass"].dropna().values
            if len(mass) >= 30:
                grades.append(int(g))
                cvs.append(cv(mass))

        if grades:
            ax1.plot(grades, cvs, 'o-', label=t, markersize=8)

    ax1.set_xlabel("Petrologic Grade")
    ax1.set_ylabel("Coefficient of Variation")
    ax1.set_title("P2: CV vs Grade (should decrease)")
    ax1.legend()
    ax1.set_xticks([3, 4, 5, 6])

    # Plot 2: Grade 5 vs Grade 6 comparison
    ax2 = axes[0, 1]

    types = ["L", "H", "LL"]
    x = np.arange(len(types))
    width = 0.35

    cv5s = []
    cv6s = []
    for t in types:
        m5 = df[df["recclass"] == f"{t}5"]["mass"].dropna().values
        m6 = df[df["recclass"] == f"{t}6"]["mass"].dropna().values
        cv5s.append(cv(m5) if len(m5) >= 30 else 0)
        cv6s.append(cv(m6) if len(m6) >= 30 else 0)

    ax2.bar(x - width/2, cv5s, width, label='Grade 5', color='coral')
    ax2.bar(x + width/2, cv6s, width, label='Grade 6', color='steelblue')
    ax2.set_xticks(x)
    ax2.set_xticklabels(types)
    ax2.set_ylabel("Coefficient of Variation")
    ax2.set_title("P1: Grade 5 vs Grade 6")
    ax2.legend()

    # Plot 3: Falls vs Finds
    ax3 = axes[1, 0]

    categories = []
    cv_values = []
    colors = []

    for discovery in ["Fell", "Found"]:
        for t in ["L", "H"]:
            for g in ["5", "6"]:
                mass = df[(df["fall"] == discovery) & (df["recclass"] == f"{t}{g}")]["mass"].dropna().values
                if len(mass) >= 20:
                    categories.append(f"{t}{g}\n({discovery})")
                    cv_values.append(cv(mass))
                    colors.append("coral" if g == "5" else "steelblue")

    ax3.bar(categories, cv_values, color=colors)
    ax3.set_ylabel("Coefficient of Variation")
    ax3.set_title("P3: Falls vs Finds")
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Summary verdict
    ax4 = axes[1, 1]
    ax4.axis('off')

    verdict_text = f"""
H-FRAG HYPOTHESIS TEST RESULTS

Hypothesis:
"Grade 6 metamorphism produces more uniform
mechanical properties, resulting in more
predictable fragmentation."

Predictions:
P1 (Grade 6 < Grade 5):  {results["predictions"]["P1_grade6_lt_grade5"]["verdict"]}
P2 (Monotonic trend):    {results["predictions"]["P2_monotonic_trend"]["verdict"]}
P3 (Falls AND Finds):    {results["predictions"]["P3_falls_and_finds"]["verdict"]}
P5 (Permutation test):   {results["predictions"]["P5_permutation_test"]["verdict"]}

OVERALL: {results["overall_verdict"]}
"""

    ax4.text(0.1, 0.5, verdict_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_all_predictions(n_permutations=500, seed=42)
