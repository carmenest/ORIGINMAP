from originmap.pipeline.visualize import visualize
from originmap.config import REPORTS

def test_visualizations_created():
    outputs = visualize()
    for path in outputs:
        assert (REPORTS / path.split("/")[-1]).exists()
