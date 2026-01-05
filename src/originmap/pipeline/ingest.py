import pandas as pd
from pathlib import Path
from originmap.config import DATA_PROCESSED, DATA_RAW

RAW_FILE = DATA_RAW / "meteorites_mbdb.csv"

def ingest_meteorites():
    if not RAW_FILE.exists():
        raise FileNotFoundError("Raw meteorite dataset not found. Run download first.")

    df = pd.read_csv(RAW_FILE)

    # Selección mínima y segura de columnas comunes
    columns = [
        "name",
        "id",
        "nametype",
        "recclass",
        "mass",
        "fall",
        "year",
        "reclat",
        "reclong",
    ]

    available_cols = [c for c in columns if c in df.columns]
    df = df[available_cols].copy()

    # Normalización básica (sin interpretación)
    if "year" in df.columns:
        # Year is already numeric (1880, 1951, etc.), just ensure it's numeric type
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "mass" in df.columns:
        df["mass"] = pd.to_numeric(df["mass"], errors="coerce")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "meteorites.parquet"

    df.to_parquet(out_path, index=False)

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "output_file": str(out_path),
    }
