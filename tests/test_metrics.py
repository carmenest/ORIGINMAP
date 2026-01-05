from originmap.pipeline.metrics import compute_metrics
from originmap.config import REPORTS

def test_metrics_outputs_created():
    info = compute_metrics()
    assert (REPORTS / "metrics_summary.json").exists()
    assert (REPORTS / "diversity_by_class.csv").exists()
