import pandas as pd
import numpy as np
from pathlib import Path
from math import log
from originmap.config import DATA_PROCESSED, REPORTS

DATASET = DATA_PROCESSED / "meteorites.parquet"

def shannon_entropy(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True)
    return -sum(p * log(p) for p in probs if p > 0)

def simpson_index(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True)
    return 1 - sum(p**2 for p in probs)

def compute_metrics():
    if not DATASET.exists():
        raise FileNotFoundError("Processed dataset not found. Run ingest first.")

    df = pd.read_parquet(DATASET)

    results = {}

    # --- Global metrics ---
    results["global"] = {
        "n_samples": len(df),
        "n_classes": df["recclass"].nunique() if "recclass" in df else None,
    }

    # --- Mass statistics ---
    if "mass" in df:
        mass = df["mass"].dropna()
        results["mass"] = {
            "mean": float(mass.mean()),
            "median": float(mass.median()),
            "std": float(mass.std()),
            "p05": float(mass.quantile(0.05)),
            "p25": float(mass.quantile(0.25)),
            "p75": float(mass.quantile(0.75)),
            "p95": float(mass.quantile(0.95)),
        }

    # --- Temporal distribution ---
    if "year" in df:
        year_counts = df["year"].dropna().value_counts().sort_index()
        results["temporal"] = {
            "min_year": int(year_counts.index.min()),
            "max_year": int(year_counts.index.max()),
        }

    # --- Diversity by meteorite class ---
    if "recclass" in df:
        diversity = []
        for cls, group in df.groupby("recclass"):
            diversity.append({
                "recclass": cls,
                "count": len(group),
                "shannon_entropy": shannon_entropy(group["name"]),
                "simpson_index": simpson_index(group["name"]),
            })

        diversity_df = pd.DataFrame(diversity)
        diversity_path = REPORTS / "diversity_by_class.csv"
        diversity_df.to_csv(diversity_path, index=False)

        results["diversity_table"] = str(diversity_path)

    # --- Save summary table ---
    summary_path = REPORTS / "metrics_summary.json"
    pd.Series(results).to_json(summary_path, indent=2)

    return {
        "summary": str(summary_path),
        "details": results,
    }
