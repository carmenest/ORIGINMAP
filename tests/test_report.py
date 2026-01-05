from originmap.pipeline.report import build_report
from originmap.config import REPORTS

def test_report_created():
    path = build_report()
    assert (REPORTS / path.split("/")[-1]).exists()
