# Meteorites Catalog with Observation Metadata

A curated version of the NASA Meteorite Landings dataset, enhanced with features that expose observational biases.

## Purpose

This dataset is designed for **methodological analysis**, not discovery. It demonstrates how scientific catalogs accumulate biases over time and how these biases can generate statistically convincing but physically spurious patterns.

## Source

- **Original**: NASA Open Data Portal - Meteorite Landings
- **Samples**: 45,716 meteorites
- **Time span**: 860 CE to 2013

## Feature Engineering

| Column | Description | Why It Matters |
|--------|-------------|----------------|
| `name` | Meteorite identifier | Antarctica specimens have systematic naming conventions |
| `mass` | Mass in grams | Raw measurement |
| `log_mass` | log(mass) | Distribution is log-normal; essential for proper analysis |
| `recclass` | Official classification | 461 unique classes |
| `petrologic_type` | Extracted type (H, L, LL, E, C, Iron, etc.) | Groups related classes |
| `petrologic_grade` | Metamorphic grade (3-7) | Indicates thermal history |
| `collection_method` | Fall (witnessed) or Find (discovered) | **Critical**: Falls are unbiased; Finds are not |
| `year` | Year recorded | Enables temporal bias analysis |
| `decade` | Decade (1970, 1980, etc.) | For aggregated temporal analysis |
| `era` | pre-1970 / antarctica-era / modern | Captures major catalog shifts |
| `antarctica_flag` | Boolean | 55% of catalog is from Antarctica |
| `mass_category` | Size bin (tiny, small, medium, large, very_large) | For stratified analysis |
| `reclat`, `reclong` | Coordinates | Geographic analysis |

## Key Biases Encoded

### 1. Collection Method Bias
- **Falls** (2.4%): Unbiased sample — we saw them fall
- **Finds** (97.6%): Biased toward larger, more visible specimens

### 2. Temporal Bias
- Pre-1970: Median mass ~4,000g (we found big ones first)
- Post-1970: Median mass ~30g (Antarctica small specimens flood catalog)

### 3. Geographic Bias
- 55% of specimens are from Antarctica
- Antarctica specimens are systematically smaller

### 4. Classification Bias
- Falls receive provisional labels (e.g., "Stone-uncl", "OC")
- Finds receive precise grades after laboratory analysis
- Class frequencies differ between Falls and Finds

## Usage Notes

1. **Do not treat this as a random sample of meteorites arriving on Earth**
2. **Always stratify by `collection_method` when studying frequencies**
3. **Use `era` to control for temporal catalog evolution**
4. **Use balanced subsampling when comparing classes with different N**

## Statistics

| Metric | Value |
|--------|-------|
| Total samples | 45,716 |
| Falls | 1,107 (2.4%) |
| Finds | 44,609 (97.6%) |
| Antarctica | 25,219 (55%) |
| Unique classes | 461 |
| Year range | 860 - 2013 |
| Mass range | 0.01g - 60,000,000g |

## License

Original data: Public Domain (NASA)
Derived features: CC-BY-4.0

## Citation

If you use this dataset for methodological demonstrations, please acknowledge that observational biases exist and were intentionally exposed through feature engineering.
