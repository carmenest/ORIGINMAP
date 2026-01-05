from originmap.config import DATA_RAW, DATA_PROCESSED, REPORTS
from originmap.utils.provenance import create_manifest

def bootstrap():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    manifest = create_manifest(
        output_dir=REPORTS,
        inputs={},
        parameters={"step": "bootstrap"}
    )

    return manifest
