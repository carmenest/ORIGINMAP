from originmap.pipeline.ingest import ingest_meteorites
from originmap.config import DATA_PROCESSED

def test_ingest_creates_parquet():
    info = ingest_meteorites()
    out = DATA_PROCESSED / "meteorites.parquet"
    assert out.exists()
    assert info["rows"] > 0
