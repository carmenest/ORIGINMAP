from originmap.pipeline.bootstrap import bootstrap
from originmap.config import REPORTS

def test_bootstrap_creates_manifest():
    manifest = bootstrap()
    assert manifest.exists()
    assert manifest.parent == REPORTS
