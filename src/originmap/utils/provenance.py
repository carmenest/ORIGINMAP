import json
from datetime import datetime
from pathlib import Path
import platform
import sys
import uuid

def create_manifest(output_dir: Path, inputs: dict, parameters: dict):
    run_id = uuid.uuid4().hex[:12]

    manifest = {
        "project": "ORIGINMAP",
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "inputs": inputs,
        "parameters": parameters,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"manifest_{run_id}.json"

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    return path
