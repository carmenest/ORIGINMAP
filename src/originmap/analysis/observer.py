"""
Automated observation system for ORIGINMAP.
Runs pipeline, detects anomalies, creates timestamped observation logs.
"""
import json
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from originmap.config import PROJECT_ROOT, REPORTS, DATA_PROCESSED
from originmap.analysis.anomalies import run_all_detections


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


NOTES_DIR = PROJECT_ROOT / "notes"
OBSERVATIONS_DIR = NOTES_DIR / "observations"
COMPARISONS_DIR = NOTES_DIR / "comparisons"


def setup_notes_structure():
    """Create notes directory structure."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    OBSERVATIONS_DIR.mkdir(exist_ok=True)
    COMPARISONS_DIR.mkdir(exist_ok=True)


def get_current_run_id() -> Optional[str]:
    """Get the most recent run_id from manifests."""
    manifests = sorted(REPORTS.glob("manifest_*.json"))
    if not manifests:
        return None
    with open(manifests[-1]) as f:
        return json.load(f).get("run_id")


def generate_observation_id() -> str:
    """Generate unique observation ID based on timestamp."""
    ts = datetime.utcnow()
    return ts.strftime("%Y%m%d_%H%M%S")


def load_previous_observations() -> list:
    """Load all previous observation files."""
    observations = []
    for f in sorted(OBSERVATIONS_DIR.glob("observation_*.json")):
        with open(f) as fp:
            observations.append(json.load(fp))
    return observations


def compute_dataset_fingerprint() -> str:
    """Compute hash of current dataset for change detection."""
    path = DATA_PROCESSED / "meteorites.parquet"
    if not path.exists():
        return "no_dataset"

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def compare_with_previous(current: Dict[str, Any], previous: list) -> Dict[str, Any]:
    """Compare current observation with previous ones."""
    comparison = {
        "is_first_run": len(previous) == 0,
        "previous_count": len(previous),
        "changes": [],
        "stable_anomalies": [],
        "new_anomalies": [],
        "resolved_anomalies": [],
    }

    if len(previous) == 0:
        return comparison

    # Get most recent previous
    prev = previous[-1]

    # Compare dataset fingerprint
    if current.get("dataset_fingerprint") != prev.get("dataset_fingerprint"):
        comparison["changes"].append({
            "type": "dataset_changed",
            "previous": prev.get("dataset_fingerprint"),
            "current": current.get("dataset_fingerprint")
        })

    # Compare anomaly counts
    prev_anomalies = set(
        f"{a['detector']}:{a['description']}"
        for a in prev.get("anomalies", [])
    )
    curr_anomalies = set(
        f"{a['detector']}:{a['description']}"
        for a in current.get("anomalies", [])
    )

    comparison["stable_anomalies"] = list(prev_anomalies & curr_anomalies)
    comparison["new_anomalies"] = list(curr_anomalies - prev_anomalies)
    comparison["resolved_anomalies"] = list(prev_anomalies - curr_anomalies)

    return comparison


def format_observation_markdown(obs: Dict[str, Any], comparison: Dict[str, Any]) -> str:
    """Format observation as markdown for human review."""
    lines = [
        f"# Observation {obs['observation_id']}",
        "",
        f"**Generated**: {obs['timestamp_utc']}",
        f"**Run ID**: {obs['run_id']}",
        f"**Dataset Fingerprint**: {obs['dataset_fingerprint']}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- Total anomalies detected: **{obs['anomaly_count']}**",
        f"- Dataset rows: {obs['dataset_shape']['rows']}",
        f"- Dataset columns: {obs['dataset_shape']['columns']}",
        "",
    ]

    # Comparison section
    if not comparison["is_first_run"]:
        lines.extend([
            "## Comparison with Previous Run",
            "",
            f"- Previous observations: {comparison['previous_count']}",
            f"- Stable anomalies: {len(comparison['stable_anomalies'])}",
            f"- New anomalies: {len(comparison['new_anomalies'])}",
            f"- Resolved anomalies: {len(comparison['resolved_anomalies'])}",
            "",
        ])

        if comparison["new_anomalies"]:
            lines.append("### New Anomalies (investigate)")
            for a in comparison["new_anomalies"]:
                lines.append(f"- {a}")
            lines.append("")

        if comparison["resolved_anomalies"]:
            lines.append("### Resolved Anomalies")
            for a in comparison["resolved_anomalies"]:
                lines.append(f"- {a}")
            lines.append("")

    # Anomalies section
    lines.extend([
        "## Anomalies Detected",
        "",
    ])

    for anom in obs.get("anomalies", []):
        sig_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(anom.get("significance", "low"), "⚪")
        lines.extend([
            f"### {sig_emoji} {anom['description']}",
            "",
            f"- **Detector**: {anom['detector']}",
            f"- **Value**: {anom['value']}",
            f"- **Expected**: {anom.get('expected', 'N/A')}",
            f"- **Significance**: {anom.get('significance', 'unknown')}",
            "",
        ])

    # Checklist
    lines.extend([
        "---",
        "",
        "## Verification Checklist",
        "",
        "- [ ] Reproducido (same result on re-run)",
        "- [ ] Estable (survives small parameter changes)",
        "- [ ] Numerico (appears in raw numbers, not just visuals)",
        "",
        "## Notes",
        "",
        "_Add observations here_",
        "",
    ])

    return "\n".join(lines)


def run_observation_cycle(run_pipeline: bool = False) -> Dict[str, Any]:
    """
    Run a complete observation cycle.

    Args:
        run_pipeline: If True, run the full pipeline before detecting anomalies.

    Returns:
        Observation results.
    """
    setup_notes_structure()

    obs_id = generate_observation_id()
    timestamp = datetime.utcnow().isoformat()

    # Optionally run pipeline
    if run_pipeline:
        from originmap.pipeline.download import download_meteorites
        from originmap.pipeline.ingest import ingest_meteorites
        from originmap.pipeline.metrics import compute_metrics
        from originmap.pipeline.visualize import visualize
        from originmap.pipeline.report import build_report

        download_meteorites()
        ingest_meteorites()
        compute_metrics()
        visualize()
        build_report()

    # Run anomaly detection
    detection_results = run_all_detections()

    # Build observation record
    observation = {
        "observation_id": obs_id,
        "timestamp_utc": timestamp,
        "run_id": get_current_run_id(),
        "dataset_fingerprint": compute_dataset_fingerprint(),
        "dataset_shape": detection_results["dataset_shape"],
        "anomaly_count": detection_results["anomaly_count"],
        "anomalies": detection_results["all_anomalies"],
        "detections": detection_results["detections"],
    }

    # Compare with previous
    previous = load_previous_observations()
    comparison = compare_with_previous(observation, previous)
    observation["comparison"] = comparison

    # Save JSON observation
    json_path = OBSERVATIONS_DIR / f"observation_{obs_id}.json"
    with open(json_path, "w") as f:
        json.dump(observation, f, indent=2, cls=NumpyEncoder)

    # Save markdown observation
    md_path = OBSERVATIONS_DIR / f"observation_{obs_id}.md"
    md_content = format_observation_markdown(observation, comparison)
    with open(md_path, "w") as f:
        f.write(md_content)

    # Save comparison if not first run
    if not comparison["is_first_run"]:
        comp_path = COMPARISONS_DIR / f"comparison_{obs_id}.json"
        with open(comp_path, "w") as f:
            json.dump(comparison, f, indent=2, cls=NumpyEncoder)

    return {
        "observation_id": obs_id,
        "json_path": str(json_path),
        "md_path": str(md_path),
        "anomaly_count": observation["anomaly_count"],
        "comparison": comparison,
    }


if __name__ == "__main__":
    result = run_observation_cycle(run_pipeline=False)
    print(f"Observation {result['observation_id']} complete")
    print(f"  Anomalies: {result['anomaly_count']}")
    print(f"  JSON: {result['json_path']}")
    print(f"  Markdown: {result['md_path']}")
