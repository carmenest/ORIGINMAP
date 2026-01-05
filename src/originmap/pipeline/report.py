import json
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from originmap.config import REPORTS

def build_report():
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Locate latest manifest
    manifests = sorted(REPORTS.glob("manifest_*.json"))
    if not manifests:
        raise FileNotFoundError("No manifest found. Run previous steps first.")

    manifest_path = manifests[-1]
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load metrics summary
    metrics_path = REPORTS / "metrics_summary.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Load diversity table
    diversity_path = REPORTS / "diversity_by_class.csv"
    diversity_rows = []
    if diversity_path.exists():
        import pandas as pd
        diversity_rows = pd.read_csv(diversity_path).to_dict(orient="records")

    # Collect figures
    figures = sorted([
        p.name for p in REPORTS.glob("*.png")
    ])

    # Jinja environment
    env = Environment(
        loader=FileSystemLoader(Path(__file__).resolve().parents[3] / "templates")
    )
    template = env.get_template("report.html.j2")

    html = template.render(
        project="ORIGINMAP",
        generated_utc=datetime.utcnow().isoformat(),
        manifest=manifest,
        metrics=metrics,
        diversity=diversity_rows,
        figures=figures,
    )

    out_path = REPORTS / f"report_{manifest['run_id']}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return str(out_path)
