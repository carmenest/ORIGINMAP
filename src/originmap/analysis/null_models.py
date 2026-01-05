"""
Null Models for Hypothesis Testing.

Null Models:
- Null-1: Global permutation (baseline)
- Null-2: Mass-bin stratified permutation
- Null-3: Mass × Time stratified permutation
- Null-4: Mass × Fall/Found stratified permutation
- Null-5: Balanced subsampling (controls for sample size)

Each null model generates a distribution of expected statistics
under the null hypothesis that class labels don't carry information
beyond the controlled confounders.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Tuple, Optional
from dataclasses import dataclass
from .stats_robust import STAT_FUNCTIONS, compute_all_stats


@dataclass
class NullResult:
    """Result from a null model test."""
    null_name: str
    stat_name: str
    observed: float
    null_mean: float
    null_std: float
    z_score: float
    p_value: float
    quantile: float
    n_permutations: int
    null_distribution: List[float]

    def to_dict(self) -> Dict:
        return {
            "null_name": self.null_name,
            "stat_name": self.stat_name,
            "observed": float(self.observed),
            "null_mean": float(self.null_mean),
            "null_std": float(self.null_std),
            "z_score": float(self.z_score),
            "p_value": float(self.p_value),
            "quantile": float(self.quantile),
            "n_permutations": self.n_permutations
        }


class NullModel:
    """Base class for null models."""

    name: str = "base"
    description: str = "Base null model"

    def __init__(self, df: pd.DataFrame, n_mass_bins: int = 10):
        """
        Initialize null model.

        Args:
            df: DataFrame with columns ['recclass', 'mass', 'year', 'fall']
            n_mass_bins: Number of bins for mass stratification
        """
        self.df = df.copy()
        self.n_mass_bins = n_mass_bins
        self._prepare_strata()

    def _prepare_strata(self):
        """Prepare stratification columns. Override in subclasses."""
        pass

    def _get_stratum_key(self, row) -> str:
        """Get stratum key for a row. Override in subclasses."""
        return "global"

    def permute(self, rng: np.random.Generator) -> pd.DataFrame:
        """
        Generate one permutation of class labels.

        Override in subclasses to implement stratified permutation.
        """
        df_perm = self.df.copy()
        df_perm["recclass"] = rng.permutation(df_perm["recclass"].values)
        return df_perm

    def compute_stat(
        self,
        df: pd.DataFrame,
        target_class: str,
        stat_fn: Callable
    ) -> float:
        """Compute statistic for a class in a (permuted) dataframe."""
        masses = df[df["recclass"] == target_class]["mass"].dropna().values
        if len(masses) < 5:
            return np.nan
        return stat_fn(masses)

    def run(
        self,
        target_class: str,
        stat_name: str = "cv",
        n_permutations: int = 500,
        seed: int = 42
    ) -> NullResult:
        """
        Run null model test for a specific class and statistic.
        """
        rng = np.random.default_rng(seed)
        stat_fn = STAT_FUNCTIONS[stat_name]

        # Observed statistic
        observed = self.compute_stat(self.df, target_class, stat_fn)

        # Null distribution
        null_dist = []
        for _ in range(n_permutations):
            df_perm = self.permute(rng)
            null_stat = self.compute_stat(df_perm, target_class, stat_fn)
            if not np.isnan(null_stat):
                null_dist.append(null_stat)

        null_dist = np.array(null_dist)

        if len(null_dist) < 10:
            return NullResult(
                null_name=self.name,
                stat_name=stat_name,
                observed=observed,
                null_mean=np.nan,
                null_std=np.nan,
                z_score=np.nan,
                p_value=np.nan,
                quantile=np.nan,
                n_permutations=n_permutations,
                null_distribution=[]
            )

        null_mean = np.mean(null_dist)
        null_std = np.std(null_dist, ddof=1)

        if null_std == 0:
            z_score = 0.0
        else:
            z_score = (observed - null_mean) / null_std

        # Two-tailed p-value
        p_value = 2 * min(
            np.mean(null_dist <= observed),
            np.mean(null_dist >= observed)
        )

        # Quantile (where observed falls in null distribution)
        quantile = np.mean(null_dist <= observed)

        return NullResult(
            null_name=self.name,
            stat_name=stat_name,
            observed=observed,
            null_mean=null_mean,
            null_std=null_std,
            z_score=z_score,
            p_value=p_value,
            quantile=quantile,
            n_permutations=n_permutations,
            null_distribution=null_dist.tolist()
        )


class Null1Global(NullModel):
    """
    Null-1: Global permutation.

    Permutes class labels globally without any stratification.
    Baseline null model - tests if class membership is random.
    """

    name = "null1"
    description = "Global permutation (baseline)"

    def permute(self, rng: np.random.Generator) -> pd.DataFrame:
        df_perm = self.df.copy()
        df_perm["recclass"] = rng.permutation(df_perm["recclass"].values)
        return df_perm


class Null2MassBins(NullModel):
    """
    Null-2: Mass-bin stratified permutation.

    Permutes class labels only within mass quantile bins.
    Controls for: mass distribution.
    """

    name = "null2"
    description = "Mass-bin stratified"

    def _prepare_strata(self):
        self.df["mass_bin"] = pd.qcut(
            self.df["mass"],
            q=self.n_mass_bins,
            labels=False,
            duplicates="drop"
        )

    def permute(self, rng: np.random.Generator) -> pd.DataFrame:
        df_perm = self.df.copy()

        for bin_val in df_perm["mass_bin"].unique():
            mask = df_perm["mass_bin"] == bin_val
            df_perm.loc[mask, "recclass"] = rng.permutation(
                df_perm.loc[mask, "recclass"].values
            )

        return df_perm


class Null3MassTime(NullModel):
    """
    Null-3: Mass × Time stratified permutation.

    Permutes class labels within (mass_bin, time_period) strata.
    Controls for: mass distribution AND collection epoch.
    """

    name = "null3"
    description = "Mass × Time stratified"

    def _prepare_strata(self):
        # Mass bins
        self.df["mass_bin"] = pd.qcut(
            self.df["mass"],
            q=self.n_mass_bins,
            labels=False,
            duplicates="drop"
        )

        # Time periods
        def time_period(year):
            if pd.isna(year):
                return "unknown"
            elif year < 1970:
                return "pre-1970"
            elif year < 1990:
                return "1970-1990"
            else:
                return "post-1990"

        self.df["time_period"] = self.df["year"].apply(time_period)
        self.df["stratum"] = (
            self.df["mass_bin"].astype(str) + "_" +
            self.df["time_period"]
        )

    def permute(self, rng: np.random.Generator) -> pd.DataFrame:
        df_perm = self.df.copy()

        for stratum in df_perm["stratum"].unique():
            mask = df_perm["stratum"] == stratum
            df_perm.loc[mask, "recclass"] = rng.permutation(
                df_perm.loc[mask, "recclass"].values
            )

        return df_perm


class Null4MassFall(NullModel):
    """
    Null-4: Mass × Fall/Found stratified permutation.

    Permutes class labels within (mass_bin, fall_type) strata.
    Controls for: mass distribution AND observation method.
    """

    name = "null4"
    description = "Mass × Fall/Found stratified"

    def _prepare_strata(self):
        # Mass bins
        self.df["mass_bin"] = pd.qcut(
            self.df["mass"],
            q=self.n_mass_bins,
            labels=False,
            duplicates="drop"
        )

        # Fall type (normalize)
        self.df["fall_type"] = self.df["fall"].fillna("Unknown").str.lower()
        self.df["stratum"] = (
            self.df["mass_bin"].astype(str) + "_" +
            self.df["fall_type"]
        )

    def permute(self, rng: np.random.Generator) -> pd.DataFrame:
        df_perm = self.df.copy()

        for stratum in df_perm["stratum"].unique():
            mask = df_perm["stratum"] == stratum
            df_perm.loc[mask, "recclass"] = rng.permutation(
                df_perm.loc[mask, "recclass"].values
            )

        return df_perm


class Null5Balanced(NullModel):
    """
    Null-5: Balanced subsampling.

    Controls for sample size by subsampling each class to a common size
    before computing statistics.

    This breaks the sample-size confound: larger classes don't have
    lower variance just because of n.
    """

    name = "null5"
    description = "Balanced subsampling"

    def __init__(
        self,
        df: pd.DataFrame,
        n_mass_bins: int = 10,
        subsample_size: int = 100
    ):
        self.subsample_size = subsample_size
        super().__init__(df, n_mass_bins)

    def _prepare_strata(self):
        # Mass bins for stratified subsampling
        self.df["mass_bin"] = pd.qcut(
            self.df["mass"],
            q=self.n_mass_bins,
            labels=False,
            duplicates="drop"
        )

    def _subsample(self, df: pd.DataFrame, target_class: str, rng: np.random.Generator) -> np.ndarray:
        """Subsample a class to fixed size, preserving mass distribution."""
        class_df = df[df["recclass"] == target_class]

        if len(class_df) < self.subsample_size:
            return class_df["mass"].values

        # Stratified subsample by mass bins
        samples = []
        bin_counts = class_df["mass_bin"].value_counts()
        total = len(class_df)

        for bin_val, count in bin_counts.items():
            bin_data = class_df[class_df["mass_bin"] == bin_val]["mass"].values
            # Proportional allocation
            n_sample = max(1, int(self.subsample_size * count / total))
            n_sample = min(n_sample, len(bin_data))
            samples.extend(rng.choice(bin_data, size=n_sample, replace=False))

        return np.array(samples[:self.subsample_size])

    def compute_stat(
        self,
        df: pd.DataFrame,
        target_class: str,
        stat_fn: Callable,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """Compute statistic on subsampled data."""
        if rng is None:
            rng = np.random.default_rng(42)

        masses = self._subsample(df, target_class, rng)
        if len(masses) < 5:
            return np.nan
        return stat_fn(masses)

    def run(
        self,
        target_class: str,
        stat_name: str = "cv",
        n_permutations: int = 500,
        seed: int = 42
    ) -> NullResult:
        """
        Run balanced null model test.

        For each permutation:
        1. Permute class labels (with mass-bin stratification)
        2. Subsample target class to fixed size
        3. Compute statistic
        """
        rng = np.random.default_rng(seed)
        stat_fn = STAT_FUNCTIONS[stat_name]

        # Observed: average over multiple subsamples
        obs_samples = []
        for _ in range(50):  # Average over 50 subsamples
            obs = self.compute_stat(self.df, target_class, stat_fn, rng)
            if not np.isnan(obs):
                obs_samples.append(obs)

        if len(obs_samples) < 10:
            return NullResult(
                null_name=self.name,
                stat_name=stat_name,
                observed=np.nan,
                null_mean=np.nan,
                null_std=np.nan,
                z_score=np.nan,
                p_value=np.nan,
                quantile=np.nan,
                n_permutations=n_permutations,
                null_distribution=[]
            )

        observed = np.mean(obs_samples)

        # Null distribution
        null_dist = []
        for _ in range(n_permutations):
            # Permute with mass-bin stratification
            df_perm = self.df.copy()
            for bin_val in df_perm["mass_bin"].unique():
                mask = df_perm["mass_bin"] == bin_val
                df_perm.loc[mask, "recclass"] = rng.permutation(
                    df_perm.loc[mask, "recclass"].values
                )

            null_stat = self.compute_stat(df_perm, target_class, stat_fn, rng)
            if not np.isnan(null_stat):
                null_dist.append(null_stat)

        null_dist = np.array(null_dist)

        if len(null_dist) < 10:
            return NullResult(
                null_name=self.name,
                stat_name=stat_name,
                observed=observed,
                null_mean=np.nan,
                null_std=np.nan,
                z_score=np.nan,
                p_value=np.nan,
                quantile=np.nan,
                n_permutations=n_permutations,
                null_distribution=[]
            )

        null_mean = np.mean(null_dist)
        null_std = np.std(null_dist, ddof=1)

        if null_std == 0:
            z_score = 0.0
        else:
            z_score = (observed - null_mean) / null_std

        p_value = 2 * min(
            np.mean(null_dist <= observed),
            np.mean(null_dist >= observed)
        )
        quantile = np.mean(null_dist <= observed)

        return NullResult(
            null_name=self.name,
            stat_name=stat_name,
            observed=observed,
            null_mean=null_mean,
            null_std=null_std,
            z_score=z_score,
            p_value=p_value,
            quantile=quantile,
            n_permutations=n_permutations,
            null_distribution=null_dist.tolist()
        )


# Registry of null models
NULL_MODELS = {
    "null1": Null1Global,
    "null2": Null2MassBins,
    "null3": Null3MassTime,
    "null4": Null4MassFall,
    "null5": Null5Balanced,
}


def get_null_model(name: str, df: pd.DataFrame, **kwargs) -> NullModel:
    """Get null model instance by name."""
    if name not in NULL_MODELS:
        raise ValueError(f"Unknown null model: {name}. Available: {list(NULL_MODELS.keys())}")
    return NULL_MODELS[name](df, **kwargs)


def parse_null_range(null_spec: str) -> List[str]:
    """
    Parse null model specification.

    Examples:
        "1-5" -> ["null1", "null2", "null3", "null4", "null5"]
        "2,4" -> ["null2", "null4"]
        "1-3,5" -> ["null1", "null2", "null3", "null5"]
    """
    result = set()

    for part in null_spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            for i in range(int(start), int(end) + 1):
                result.add(f"null{i}")
        else:
            result.add(f"null{int(part)}")

    return sorted(result, key=lambda x: int(x[4:]))
