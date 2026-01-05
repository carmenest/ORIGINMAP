import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from originmap.config import DATA_PROCESSED, REPORTS

DATASET = DATA_PROCESSED / "meteorites.parquet"

def visualize():
    if not DATASET.exists():
        raise FileNotFoundError("Processed dataset not found. Run ingest first.")

    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATASET)

    outputs = []

    # --- Mass distribution ---
    if "mass" in df:
        mass = df["mass"].dropna()
        plt.figure(figsize=(8, 5))
        plt.hist(mass, bins=50)
        plt.yscale("log")
        plt.xlabel("Mass (g)")
        plt.ylabel("Frequency (log scale)")
        plt.tight_layout()
        path = REPORTS / "mass_distribution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        outputs.append(str(path))

    # --- Temporal distribution ---
    if "year" in df:
        year_counts = df["year"].dropna().value_counts().sort_index()
        plt.figure(figsize=(10, 5))
        plt.plot(year_counts.index, year_counts.values)
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.tight_layout()
        path = REPORTS / "temporal_distribution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        outputs.append(str(path))

    # --- Diversity plots ---
    diversity_csv = REPORTS / "diversity_by_class.csv"
    if diversity_csv.exists():
        div = pd.read_csv(diversity_csv)

        # Shannon
        plt.figure(figsize=(10, 6))
        plt.bar(div["recclass"], div["shannon_entropy"])
        plt.xticks(rotation=90)
        plt.ylabel("Shannon entropy")
        plt.tight_layout()
        path = REPORTS / "diversity_shannon.png"
        plt.savefig(path, dpi=150)
        plt.close()
        outputs.append(str(path))

        # Simpson
        plt.figure(figsize=(10, 6))
        plt.bar(div["recclass"], div["simpson_index"])
        plt.xticks(rotation=90)
        plt.ylabel("Simpson index")
        plt.tight_layout()
        path = REPORTS / "diversity_simpson.png"
        plt.savefig(path, dpi=150)
        plt.close()
        outputs.append(str(path))

    return outputs
