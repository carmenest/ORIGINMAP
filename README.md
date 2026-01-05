# ORIGINMAP

**Computational Laboratory for Meteorite Analysis**

A reproducible pipeline that systematically tests hypotheses about meteorite population structure using progressively stringent null models.

## Key Finding

**Apparent mass distribution structure in meteorite classes is a sample size artifact.**

When balanced subsampling is applied:
- All "significant" patterns disappear
- No class shows genuine structure
- The catalog reflects observation history, not cosmic reality

## Experiments

| Experiment | Question | Result |
|------------|----------|--------|
| O-Δ1–O-Δ5 | Mass heterogeneity in classes? | Artifact of sample size |
| O-Δ6 | Grade 5 = heterogeneity peak? | Falsified globally |
| O-Δ7 | Stability regimes exist? | Collapse under Null-5 |
| O-Δ8 | At what N does structure emerge? | Never (with balanced sampling) |
| O-Δ9 | Catalog reflects universe? | No — reflects observation history |
| O-Δ10 | Falls show genuine patterns? | Only classification bias |

## Methodology

### Null Model Battery

| Null | Description | Controls For |
|------|-------------|--------------|
| Null-1 | Global permutation | Baseline |
| Null-2 | Mass-bin stratified | Mass distribution |
| Null-3 | Mass × Time stratified | Temporal bias |
| Null-4 | Mass × Fall/Found stratified | Collection method |
| Null-5 | Balanced subsampling | **Sample size** |

### Statistics

- **CV**: Coefficient of variation
- **varlog**: Variance of log(mass)
- **MAD/median**: Robust dispersion

## Installation

```bash
git clone https://github.com/carmenest/ORIGINMAP.git
cd ORIGINMAP
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Full pipeline
originmap init
originmap download
originmap ingest
originmap metrics

# Hypothesis tests
originmap battery --null 1-5 --stats cv,varlog,mad
originmap grade-curve --perm 1000
originmap stability-clusters --clusters 5
originmap sample-threshold --perm 200
originmap temporal
originmap falls
```

## Project Structure

```
src/originmap/
├── analysis/
│   ├── stats_robust.py          # CV, varlog, MAD
│   ├── null_models.py           # Null-1 through Null-5
│   ├── hypothesis_battery.py    # Battery orchestrator
│   ├── hypothesis_grade_curve.py
│   ├── hypothesis_stability_clustering.py
│   ├── hypothesis_sample_threshold.py
│   ├── hypothesis_temporal.py
│   └── hypothesis_falls.py
├── pipeline/
│   ├── download.py
│   ├── ingest.py
│   ├── metrics.py
│   └── visualize.py
└── cli.py
```

## Key Insights

1. **Balanced subsampling is essential** — Without it, sample size confounds all conclusions

2. **Multiple null models required** — Single null model → false positives

3. **Catalogs ≠ Samples** — They are historical documents with embedded biases:
   - Pre-1970 median mass: 4,190g
   - Post-1970 median mass: 28g
   - Antarctica: 55% of catalog

4. **Falls ≠ Finds** — Classification practices differ (χ² = 1880)

## Falsifiability

| Conclusion | Would Be Falsified If... |
|------------|--------------------------|
| Structure is artifact | Any class p < 0.05 under Null-5 |
| Catalog reflects observation | H-TEMP-1,2,3 all rejected |
| Falls confirm no structure | H-FALL-1 detected structure |

## Dataset

- **Source**: NASA Meteorite Landings
- **Samples**: 45,716
- **Classes**: 461
- **Falls**: 1,107 (2.4%)
- **Finds**: 44,609 (97.6%)

## Related

- **Kaggle Dataset**: [Meteorites Catalog with Observation Metadata](https://www.kaggle.com/datasets/caresment/meteorites-catalog-with-observation-metadata)
- **Kaggle Notebook**: [When Big Data Lies](https://www.kaggle.com/code/caresment/when-big-data-lies)

## License

MIT

## Citation

If you use this methodology, please cite:

```
ORIGINMAP: Demonstrating how apparent structure in scientific catalogs
disappears under proper null models and balanced subsampling.
```
