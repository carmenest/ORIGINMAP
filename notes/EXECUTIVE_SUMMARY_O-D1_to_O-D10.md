# ORIGINMAP: Executive Summary

## Experiments O-Δ1 through O-Δ10

**Project**: ORIGINMAP — Computational Laboratory for Meteorite Analysis
**Date**: 2026-01-05
**Dataset**: NASA Meteorite Landings (45,716 samples, 461 classes)

---

## Overview

ORIGINMAP is a reproducible computational pipeline that systematically tested hypotheses about meteorite population structure using progressively stringent null models and unbiased subsamples.

**Core Questions**:
1. Do meteorite classes exhibit intrinsic constraints on mass distribution?
2. Does the catalog reflect the universe or human observation history?
3. Do Falls (unbiased sample) reveal genuine physical patterns?

---

## Experiment Timeline

```
O-Δ1 → O-Δ2 → O-Δ3 → O-Δ4/O-Δ5 → O-Δ6 → O-Δ7 → O-Δ8 → O-Δ9 → O-Δ10
  │       │       │         │         │       │       │       │       │
  │       │       │         │         │       │       │       │       └─ Falls-only analysis
  │       │       │         │         │       │       │       └─ Temporal dynamics
  │       │       │         │         │       │       └─ Sample size threshold
  │       │       │         │         │       └─ Stability clustering
  │       │       │         │         └─ Grade curvature test
  │       │       │         └─ Full null battery
  │       │       └─ Stratified null (mass bins)
  │       └─ Mass heterogeneity test
  └─ Initial diversity test (degenerate)
```

---

## Experiment Details

### O-Δ1: Diversity Test (Abandoned)
- **Objective**: Test if class diversity differs from random
- **Result**: Degenerate — `name` is unique identifier
- **Status**: Abandoned, pivoted to O-Δ2

### O-Δ2: Mass Heterogeneity Test
- **Objective**: Test if mass variance within classes differs from random
- **Method**: Global permutation, CV statistic
- **Result**: 10 classes significant (L6, H6, Ureilite, etc.)
- **Status**: Promising, but needed harder null model

### O-Δ3: Stratified Null (Mass Bins)
- **Objective**: Control for mass distribution in null model
- **Method**: Permute only within mass quantile bins
- **Result**: L6, H6, Ureilite survive
- **Status**: Structure persists under mass control

### O-Δ4/O-Δ5: Full Null Battery
- **Objective**: Test against multiple confounders
- **Method**:
  - Null-1: Global permutation
  - Null-2: Mass-bin stratified
  - Null-3: Mass × Time stratified
  - Null-4: Mass × Fall/Found stratified
  - Null-5: Balanced subsampling
- **Result**:
  - With Null-1 to Null-4: L6 survives all
  - With Null-5: NO class survives
- **Status**: Critical finding — Null-5 destroys all structure

### O-Δ6: Grade Curvature Test (H-FRAG-2)
- **Objective**: Test if heterogeneity peaks at Grade 5
- **Hypothesis**: H(5) > H(4) AND H(5) > H(6)
- **Method**: Permutation test on grade labels
- **Result**:
  - Global: FALSIFIED (Grade 4 is maximum, not 5)
  - By type: L-type shows effect (p=0.008 for MAD)
- **Status**: Universal hypothesis falsified; type-dependent structure exists

### O-Δ7: Stability Clustering
- **Objective**: Identify stability regimes across classes
- **Method**: Hierarchical clustering on survival patterns
- **Result**: 5 distinct regimes identified
  - Regime 3 (53% survival): L6, H6, H4, Ureilite
  - Regime 1 (1% survival): Iron, CO3, CV3, etc.
- **Key Finding**: Null-5 centroid = 0 for all regimes
- **Status**: Structure is regime-dependent, but collapses under Null-5

### O-Δ8: Sample Size Threshold
- **Objective**: At what N does structure emerge?
- **Method**: Sweep subsample sizes 30-300, track significance
- **Result**:
  ```
  L6:       NO structure at any N
  H6:       NO structure at any N
  H4:       NO structure at any N
  H5:       NO structure at any N
  L5:       NO structure at any N
  LL6:      NO structure at any N
  Ureilite: Structure at N=300 (borderline)
  ```
- **Status**: CRITICAL — All "structure" is sample size artifact

### O-Δ9: Catalog Archaeology (Temporal Dynamics)
- **Objective**: Does the catalog reflect the universe or observation history?
- **Hypotheses**:
  - H-TEMP-1: Classes appear in "waves" (not uniformly)
  - H-TEMP-2: Mean discovery mass decreases over time
  - H-TEMP-3: Antarctica distorts the catalog post-1970
- **Method**:
  - Era segmentation (Classical, Modern, Antarctica, Satellite)
  - KS test for uniformity of class first-appearance
  - Spearman correlation for mass trend
  - KS test for mass distribution shift pre/post-1970
- **Result**:
  | Hypothesis | Test | Statistic | p-value | Result |
  |------------|------|-----------|---------|--------|
  | H-TEMP-1 | KS uniformity | — | <10⁻¹⁴³ | **SUPPORTED** |
  | H-TEMP-2 | Spearman | r = -0.621 | 0.0016 | **SUPPORTED** |
  | H-TEMP-3 | KS pre/post | — | <10⁻³⁰⁰ | **SUPPORTED** |
- **Key Metrics**:
  - Pre-1970 median mass: 4,190g
  - Post-1970 median mass: 28g (150x reduction)
  - Wave decades: 1850, 1860, 1910, 1920, 1930, 1940, 1960, 1970, 1980, 1990, 2000
- **Status**: All temporal hypotheses SUPPORTED — catalog reflects observation history

### O-Δ10: Falls-Only Analysis (Unbiased Sample)
- **Objective**: Test if Falls (witnessed meteorites) reveal genuine patterns
- **Rationale**: Falls have no collection bias (size, visibility, location)
- **Hypotheses**:
  - H-FALL-1: Falls show genuine mass structure (survives Null-5)
  - H-FALL-2: Class proportions differ between Falls and Finds
  - H-FALL-3: Falls show seasonality (orbital signature)
- **Method**:
  - Balanced subsampling permutation test (H-FALL-1)
  - Chi-square + Fisher exact with BH correction (H-FALL-2)
  - Rayleigh test for circular uniformity (H-FALL-3)
- **Result**:
  | Hypothesis | Test | Statistic | p-value | Result |
  |------------|------|-----------|---------|--------|
  | H-FALL-1 | Permutation | CV spread | 0.145 | NO_STRUCTURE |
  | H-FALL-2 | Chi-square | χ² = 1880 | <10⁻³⁰⁰ | **STRONG_DIFFERENCE** |
  | H-FALL-3 | Rayleigh | — | N/A | No month data |
- **Key Discovery — Class Enrichment in Falls**:
  | Class | Falls/Finds Ratio | Interpretation |
  |-------|-------------------|----------------|
  | Stone-uncl | 161x | Provisional classification at fall |
  | OC | 17.6x | Generic "ordinary chondrite" label |
  | L | 10x | Ungraded L-type at fall |
  | Eucrite-mmict | 8x | Over-represented in witnessed falls |
  | H | 7.4x | Ungraded H-type at fall |
  | LL5 | 0.28x | Under-represented (classified later) |
  | H4 | 0.48x | Under-represented (classified later) |
- **Mass Comparison**:
  - Falls median: 2,800g
  - Finds median: 31g
  - Ratio: **91.7x** (Falls are much larger)
- **Status**: H-FALL-2 reveals **classification bias** between Falls and Finds

---

## Hypothesis Status Summary

### Mass Structure Hypotheses

| ID | Hypothesis | Test | Falsifiable Prediction | Result |
|----|------------|------|------------------------|--------|
| H-CONS | Mass distributions have intrinsic constraints | Null-5 permutation | Classes should show low p-values under balanced subsampling | ✗ **FALSIFIED** |
| H-FRAG | Grade 6 → uniform fragmentation | CV comparison | CV(Grade 6) < CV(other grades) | ✗ **FALSIFIED** |
| H-FRAG-2 | Grade 5 is heterogeneity maximum (universal) | Permutation test | H(5) > H(4) AND H(5) > H(6) for all types | ✗ **FALSIFIED** |
| H-FRAG-3 | Grade 5 maximum is type-dependent | Permutation by type | L-type shows interior maximum | ◐ **PARTIAL** (p=0.008) |

### Temporal Hypotheses

| ID | Hypothesis | Test | Falsifiable Prediction | Result |
|----|------------|------|------------------------|--------|
| H-TEMP-1 | Classes appear in waves | KS uniformity | First-appearance years deviate from uniform | ✓ **SUPPORTED** (p<10⁻¹⁴³) |
| H-TEMP-2 | Mass decreases over time | Spearman correlation | Negative correlation between decade and median mass | ✓ **SUPPORTED** (r=-0.621) |
| H-TEMP-3 | Antarctica distorts catalog | KS pre/post 1970 | Mass distributions differ pre/post-1970 | ✓ **SUPPORTED** (p≈0) |

### Falls Hypotheses

| ID | Hypothesis | Test | Falsifiable Prediction | Result |
|----|------------|------|------------------------|--------|
| H-FALL-1 | Falls show genuine mass structure | Null-5 on Falls | CV spread exceeds null distribution | ✗ **FALSIFIED** (p=0.145) |
| H-FALL-2 | Class proportions differ Falls/Finds | Chi-square + Fisher | Significant enrichment/depletion | ✓ **SUPPORTED** (χ²=1880) |
| H-FALL-3 | Falls show seasonality | Rayleigh test | Mean resultant length > 0 | ? **NO DATA** |

---

## Key Findings

### 1. The Null-5 Collapse (O-Δ5, O-Δ8)

When balanced subsampling is applied (equal N per class), ALL apparent mass structure disappears:

```
Before Null-5:  L6, H6, H4 show significant structure
After Null-5:   ZERO classes show significant structure (including Falls)
```

**Interpretation**: The "tight mass distributions" in L6/H6 were statistical artifacts of large sample size, not physical constraints.

### 2. Catalog Reflects Observation History (O-Δ9)

Three independent tests confirm the catalog is shaped by human activity:

| Era | Years | N | Median Mass | New Classes |
|-----|-------|---|-------------|-------------|
| Classical | pre-1900 | 730 | 5,400g | 89 |
| Modern | 1900-1969 | 1,566 | 3,845g | 68 |
| Antarctica | 1970-1999 | 23,410 | 20g | 191 |
| Satellite | 2000+ | 19,720 | 46g | 114 |

**Interpretation**: We found the big meteorites first. Antarctica flooded the catalog with small specimens post-1970.

### 3. Classification Bias in Falls vs Finds (O-Δ10)

Falls and Finds have dramatically different class proportions:

- **Stone-uncl 161x enriched in Falls**: Provisional classification at time of fall
- **OC, L, H enriched in Falls**: Generic labels before laboratory analysis
- **LL5, H4 depleted in Falls**: Precise grades assigned later to Finds

**Interpretation**: Classification practices differ between witnessed falls and later finds. This biases any frequency study.

### 4. Falls Confirm No Genuine Mass Structure (O-Δ10)

Even in the unbiased Falls sample (N=1,107), no class shows genuine mass heterogeneity structure under balanced subsampling (p=0.145).

**Interpretation**: The null hypothesis of random mass distribution within classes cannot be rejected, even with unbiased data.

---

## Falsifiability Framework

### What Would Have Falsified Our Conclusions?

| Conclusion | Would Be Falsified If... |
|------------|--------------------------|
| "Structure is sample size artifact" | Any class showed p<0.05 under Null-5 at N≤100 |
| "Catalog reflects observation history" | H-TEMP-1,2,3 all rejected (p>0.05) |
| "Falls confirm no structure" | H-FALL-1 showed STRUCTURE_DETECTED |
| "Classification bias exists" | H-FALL-2 showed NO_DIFFERENCE (p>0.05) |

### Tests That Could Still Falsify

1. **Isotopic data**: If O-isotope ratios cluster within classes beyond random, genuine structure exists
2. **Paired meteorites**: If known parent-body groups (HED, SNC) show mass constraints, physical processes matter
3. **Fresh falls with full analysis**: If newly witnessed falls (with immediate lab work) show structure, the pattern is real

---

## Methodological Contributions

### Validated Principles

1. **Balanced subsampling is essential**
   - Without it, sample size confounds all conclusions
   - Any study of heterogeneity MUST control for N

2. **Multiple null models required**
   - Single null model → false positives
   - Battery approach reveals what controls survive

3. **Falls ≠ Finds**
   - Class proportions differ dramatically (χ²=1880)
   - Frequency studies must account for collection method

4. **Catalog is not the universe**
   - Temporal biases dominate (91.7x mass difference Falls/Finds)
   - Geographic biases dominate (Antarctica = 51% of catalog)

### Anti-Patterns Identified

1. Trust p-values from unbalanced comparisons
2. Interpret single-null results as definitive
3. Assume "large N = reliable" (often opposite)
4. Treat Falls and Finds as equivalent samples
5. Ignore temporal evolution of the catalog

---

## Technical Infrastructure

### Pipeline Commands

```bash
# Full data pipeline
originmap init
originmap download
originmap ingest
originmap metrics
originmap visualize
originmap report

# Hypothesis tests
originmap battery --null 1-5 --bins 8,10,12,20 --stats cv,varlog,mad
originmap grade-curve --perm 1000 --bootstrap 500
originmap stability-clusters --clusters 5
originmap sample-threshold --sizes 30,50,75,100,150,200,300
originmap temporal
originmap falls
```

### Key Modules

```
src/originmap/analysis/
├── stats_robust.py              # CV, varlog, MAD statistics
├── null_models.py               # Null-1 through Null-5
├── hypothesis_battery.py        # Battery orchestrator
├── hypothesis_grade_curve.py    # O-Δ6 grade curvature
├── hypothesis_stability_clustering.py  # O-Δ7 clustering
├── hypothesis_sample_threshold.py      # O-Δ8 threshold analysis
├── hypothesis_temporal.py       # O-Δ9 catalog archaeology
└── hypothesis_falls.py          # O-Δ10 falls-only analysis
```

### Outputs Generated

```
reports/
├── battery_*_summary.json
├── battery_*_results.csv
├── O-D6_*.csv/json/png
├── O-D7_*.csv/json/png
├── O-D8_*.csv/json/png
├── O-D9_temporal_results.json
├── O-D9_era_comparison.csv
├── O-D9_temporal_analysis.png
├── O-D10_falls_results.json
├── O-D10_class_enrichment.csv
├── O-D10_falls_analysis.png
└── manifest_*.json

notes/observations/
├── observation_O-D6_*.md
├── observation_O-D7_*.md
├── observation_O-D8_*.md
├── observation_O-D9_*.md
└── observation_O-D10_*.md
```

---

## Conclusions

### Primary Results

1. **No genuine mass structure exists in meteorite classes**
   - All apparent structure is sample size artifact
   - Confirmed in both full catalog (O-Δ8) and unbiased Falls (O-Δ10)

2. **The catalog reflects human observation, not the universe**
   - Mass decreases over time (we found big ones first)
   - Antarctica dominates post-1970 (51% of catalog)
   - Classification happens in "fashion waves"

3. **Classification practices bias the data**
   - Falls get provisional labels (Stone-uncl, OC, L, H)
   - Finds get precise grades (LL5, H4, L5)
   - Frequency studies must account for this

### Contribution

ORIGINMAP demonstrates that **rigorous null model design and sample stratification are essential** for meteorite population studies. The standard approach of comparing raw statistics across classes leads to systematic false positives driven by:

- Sample size differences
- Temporal collection biases
- Geographic collection biases
- Classification practice differences

### Open Questions

1. Do isotopic/geochemical properties show genuine clustering?
2. Does seasonality exist in Falls (requires month data)?
3. Do known parent-body groups (HED, SNC, lunar) show constraints?

---

## Reproducibility

All experiments are fully reproducible:

```bash
cd /home/carmenia/originmap
source .venv/bin/activate

# Reproduce full analysis
python -m originmap.cli battery --null 1-5 --stats cv,varlog,mad --n 500
python -m originmap.cli grade-curve --perm 1000
python -m originmap.cli stability-clusters --clusters 5
python -m originmap.cli sample-threshold --perm 200
python -m originmap.cli temporal
python -m originmap.cli falls
```

Seeds: 42 (default)
Platform: Linux/WSL2
Python: 3.12.3

---

*Generated by ORIGINMAP*
*2026-01-05*
