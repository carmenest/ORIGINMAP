import requests
from pathlib import Path
from originmap.config import DATA_RAW
from originmap.utils.hashing import sha256

# Mirror del dataset NASA Meteorite Landings (original: data.nasa.gov)
# Fuente original ha sido deprecada, usando mirror público
METEORITE_URL = "https://raw.githubusercontent.com/pylablanche/Kaggle_Meteorites/master/meteorite-landings.csv"

def download_meteorites():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = DATA_RAW / "meteorites_mbdb.csv"

    r = requests.get(METEORITE_URL, timeout=60)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)

    file_hash = sha256(out_path)

    return {
        "file": str(out_path),
        "sha256": file_hash,
        "source_url": METEORITE_URL,
    }
