"""
Create curated meteorite dataset with observation metadata.
This demonstrates feature engineering with awareness of observational bias.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load raw processed data
DATA_PATH = Path("/home/carmenia/originmap/data/processed/meteorites.parquet")
OUTPUT_DIR = Path("/home/carmenia/originmap/kaggle/dataset")

df = pd.read_parquet(DATA_PATH)

# ════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

# 1. Log mass (fundamental for analysis)
df["log_mass"] = np.log(df["mass"].replace(0, np.nan))

# 2. Extract petrologic type (H, L, LL, E, C, etc.)
def extract_type(recclass):
    if pd.isna(recclass):
        return "Unknown"
    rc = str(recclass).upper()
    if rc.startswith("LL"):
        return "LL"
    elif rc.startswith("L") and not rc.startswith("LO") and not rc.startswith("LU"):
        return "L"
    elif rc.startswith("H") and not rc.startswith("HO"):
        return "H"
    elif rc.startswith("E"):
        return "E"
    elif rc.startswith("C"):
        return "C"
    elif "IRON" in rc:
        return "Iron"
    elif "ACHON" in rc or "EUCR" in rc or "DIOG" in rc or "HOWA" in rc:
        return "Achondrite"
    elif "PALL" in rc or "MESO" in rc:
        return "Stony-Iron"
    else:
        return "Other"

df["petrologic_type"] = df["recclass"].apply(extract_type)

# 3. Extract petrologic grade (3, 4, 5, 6, 7)
def extract_grade(recclass):
    if pd.isna(recclass):
        return np.nan
    rc = str(recclass)
    # Look for single digit grade
    for char in rc:
        if char in "34567":
            return int(char)
    return np.nan

df["petrologic_grade"] = df["recclass"].apply(extract_grade)

# 4. Era classification (critical for understanding bias)
def assign_era(year):
    if pd.isna(year):
        return "unknown"
    year = int(year)
    if year < 1970:
        return "pre-1970"
    elif year < 2000:
        return "antarctica-era"
    else:
        return "modern"

df["era"] = df["year"].apply(assign_era)

# 5. Antarctica flag (inferred from naming convention)
def is_antarctica(name):
    if pd.isna(name):
        return False
    name = str(name).upper()
    # Common Antarctica prefixes
    antarctica_prefixes = [
        "ALH", "ALHA", "EET", "EETA", "LEW", "LEWA", "MAC", "MET", "META",
        "MIL", "PCA", "QUE", "RKP", "TIL", "WSG", "GRA", "GRO", "LAP",
        "LAR", "MCY", "STE", "PRE", "BTN", "DOM", "HOW", "ILD", "LON",
        "OTT", "PAT", "PGP", "SAN", "SCO", "TEN", "WIS", "Y-", "Y7", "Y8", "Y9",
        "A-", "A8", "A9", "B-", "YAMATO", "ASUKA"
    ]
    for prefix in antarctica_prefixes:
        if name.startswith(prefix):
            return True
    return False

df["antarctica_flag"] = df["name"].apply(is_antarctica)

# 6. Collection method (Fall = witnessed, Find = discovered later)
df["collection_method"] = df["fall"].fillna("Unknown")

# 7. Mass category (for stratified analysis)
def mass_category(mass):
    if pd.isna(mass) or mass <= 0:
        return "unknown"
    elif mass < 10:
        return "tiny (<10g)"
    elif mass < 100:
        return "small (10-100g)"
    elif mass < 1000:
        return "medium (100g-1kg)"
    elif mass < 10000:
        return "large (1-10kg)"
    else:
        return "very_large (>10kg)"

df["mass_category"] = df["mass"].apply(mass_category)

# 8. Decade (for temporal analysis)
df["decade"] = (df["year"] // 10 * 10).astype("Int64")

# ════════════════════════════════════════════════════════════════════════════
# SELECT AND ORDER COLUMNS
# ════════════════════════════════════════════════════════════════════════════

columns_to_export = [
    "name",
    "mass",
    "log_mass",
    "recclass",
    "petrologic_type",
    "petrologic_grade",
    "collection_method",
    "year",
    "decade",
    "era",
    "antarctica_flag",
    "mass_category",
    "reclat",
    "reclong",
]

# Filter to columns that exist
columns_to_export = [c for c in columns_to_export if c in df.columns]

df_curated = df[columns_to_export].copy()

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save as CSV
csv_path = OUTPUT_DIR / "meteorites_curated.csv"
df_curated.to_csv(csv_path, index=False)

print(f"Dataset saved to: {csv_path}")
print(f"Shape: {df_curated.shape}")
print(f"\nColumns:")
for col in df_curated.columns:
    non_null = df_curated[col].notna().sum()
    print(f"  {col}: {non_null:,} non-null")

print(f"\nEra distribution:")
print(df_curated["era"].value_counts())

print(f"\nCollection method:")
print(df_curated["collection_method"].value_counts())

print(f"\nAntarctica flag:")
print(df_curated["antarctica_flag"].value_counts())

print(f"\nPetrologic type:")
print(df_curated["petrologic_type"].value_counts().head(10))
