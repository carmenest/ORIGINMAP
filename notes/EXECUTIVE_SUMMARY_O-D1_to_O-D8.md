# ORIGINMAP: Executive Summary

## Experiments O-Δ1 through O-Δ8

**Project**: ORIGINMAP — Computational Laboratory for Meteorite Analysis
**Date**: 2026-01-05
**Dataset**: NASA Meteorite Landings (45,716 samples, 461 classes)

---

## Overview

ORIGINMAP is a reproducible computational pipeline that systematically tested hypotheses about mass distribution structure in meteorite classes using progressively stringent null models.

**Core Question**: Do meteorite classes exhibit intrinsic constraints on mass distribution, or is observed structure a statistical artifact?

---

## Experiment Timeline

```
O-Δ1 → O-Δ2 → O-Δ3 → O-Δ4/O-Δ5 → O-Δ6 → O-Δ7 → O-Δ8
  │       │       │         │         │       │       │
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

---

## Key Findings

### 1. The Null-5 Collapse

When balanced subsampling is applied (equal N per class), ALL apparent structure disappears:

```
Before Null-5:  L6, H6, H4 show significant structure
After Null-5:   ZERO classes show significant structure
```

**Interpretation**: The "tight mass distributions" in L6/H6 were statistical artifacts of large sample size, not physical constraints.

### 2. Grade-Dependent Heterogeneity is Type-Specific

- Global pattern: Grade 4 > Grade 5 (not Grade 5 peak)
- L-type only: Grade 5 shows interior maximum (p=0.008)
- H-type: No interior maximum
- LL-type: Mixed results

### 3. Stability Regimes Exist but Are Superficial

Five regimes identified, but they reflect:
- Sample size gradients (large N → apparent stability)
- Statistic sensitivity (varlog vs CV vs MAD)
- NOT physical properties

---

## Hypothesis Status

| Hypothesis | Description | Status |
|------------|-------------|--------|
| H-CONS | Mass distributions have intrinsic constraints | ✗ FALSIFIED |
| H-FRAG | Grade 6 metamorphism → uniform fragmentation | ✗ FALSIFIED |
| H-FRAG-2 | Grade 5 is heterogeneity maximum (universal) | ✗ FALSIFIED |
| H-FRAG-3 | Grade 5 maximum is type-dependent | ◐ PARTIAL (L-type only) |

---

## Methodological Contributions

### What We Validated

1. **Balanced subsampling is essential**
   - Without it, sample size confounds all conclusions
   - Any study of mass heterogeneity MUST control for N

2. **Multiple null models are required**
   - Single null model → false positives
   - Battery approach reveals what controls survive

3. **Multiple statistics are required**
   - CV, varlog, MAD measure different aspects
   - "Structure" in one statistic may not exist in others

4. **Permutation tests over parametric tests**
   - No distributional assumptions
   - Exact control of confounders

### What We Learned NOT to Do

1. Trust p-values from unbalanced comparisons
2. Interpret single-null results as definitive
3. Assume "large N = reliable" (it's often the opposite)

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
```

### Key Modules

```
src/originmap/analysis/
├── stats_robust.py              # CV, varlog, MAD statistics
├── null_models.py               # Null-1 through Null-5
├── hypothesis_battery.py        # Battery orchestrator
├── hypothesis_grade_curve.py    # O-Δ6 grade curvature
├── hypothesis_stability_clustering.py  # O-Δ7 clustering
└── hypothesis_sample_threshold.py      # O-Δ8 threshold analysis
```

### Outputs Generated

```
reports/
├── battery_*_summary.json
├── battery_*_results.csv
├── battery_*_survival.csv
├── O-D6_*.csv/json/png
├── O-D7_*.csv/json/png
├── O-D8_*.csv/json/png
└── manifest_*.json

notes/observations/
├── observation_O-D2_*.md
├── observation_O-D6_*.md
├── observation_O-D7_*.md
├── observation_O-D8_*.md
└── observation_BATTERY_*.md
```

---

## Conclusion

### Primary Result

**The apparent mass distribution structure in meteorite classes is a sample size artifact.**

When balanced subsampling is applied:
- L6, H6, H4, Ureilite lose significance
- No class shows genuine structure
- All hypotheses about intrinsic constraints are falsified

### Secondary Result

**Type-dependent grade effects exist but are weak.**

L-type chondrites show a Grade 5 heterogeneity peak, but:
- Effect is marginal (p=0.05-0.008)
- Does not survive balanced subsampling
- May be a weaker form of the same artifact

### Contribution

ORIGINMAP demonstrates that **rigorous null model design is essential** for meteorite population studies. The standard approach of comparing raw statistics across classes leads to systematic false positives driven by sample size differences.

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
```

Seeds: 42 (default)
Platform: Linux/WSL2
Python: 3.12.3

---

*Generated by ORIGINMAP*
*2026-01-05*
