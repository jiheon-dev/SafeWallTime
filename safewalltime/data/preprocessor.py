from __future__ import annotations

"""UARP preprocessor: feature extraction and MinMax scaling.
"""

from typing import Optional

import numpy as np

from .base import JobTrace, Preprocessor





DEFAULT_FEATURE_COLS = [
    "job_id",
    "submit_time",
    "user_id",
    "queue",
    "num_procs",
    "cores_requested",
    "walltime",
    "memory",
]


class UARPPreprocessor(Preprocessor):
    """MinMax-scaling preprocessor matching the UARP paper.

    Args:
        feature_cols: Which DataFrame columns to use as model features.
                      Default: the six features from the paper.
        target_col: Column name for the prediction target. Default: ``"runtime"``.
    """

    def __init__(
        self,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "runtime",
    ) -> None:
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.target_col = target_col


        self._feature_names: list[str] = []
        self._feature_min: Optional[np.ndarray] = None
        self._feature_max: Optional[np.ndarray] = None
        self._target_min: Optional[float] = None
        self._target_max: Optional[float] = None





    def fit_transform(self, trace: JobTrace) -> tuple[np.ndarray, np.ndarray]:

        available = [c for c in self.feature_cols if c in trace.df.columns]
        if not available:
            raise ValueError(
                f"None of the requested feature columns {self.feature_cols} "
                f"are present in the trace (columns: {list(trace.df.columns)})"
            )
        self._feature_names = available

        X_raw = trace.df[available].values.astype(np.float64)
        y_raw = trace.df[self.target_col].values.astype(np.float64)



        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self._feature_min = np.nanmin(X_raw, axis=0)
            self._feature_max = np.nanmax(X_raw, axis=0)
        all_nan_cols = [
            col for col, mn in zip(available, self._feature_min) if np.isnan(mn)
        ]
        if all_nan_cols:
            import logging
            logging.getLogger(__name__).warning(
                "All-NaN feature columns (will be ignored in scaling): %s",
                all_nan_cols,
            )
        self._target_min = float(np.nanmin(y_raw))
        self._target_max = float(np.nanmax(y_raw))

        X = self._scale_features(X_raw)
        y = self._scale_target(y_raw)
        return X, y

    def transform(self, trace: JobTrace) -> tuple[np.ndarray, np.ndarray]:
        if self._feature_min is None:
            raise RuntimeError("Preprocessor must be fit before transform.")
        missing = [c for c in self._feature_names if c not in trace.df.columns]
        if missing:
            raise ValueError(
                f"Test trace is missing feature columns that were present "
                f"during fit: {missing}"
            )
        X_raw = trace.df[self._feature_names].values.astype(np.float64)
        y_raw = trace.df[self.target_col].values.astype(np.float64)
        X = self._scale_features(X_raw)
        y = self._scale_target(y_raw)
        return X, y

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        if self._target_min is None:
            raise RuntimeError("Preprocessor must be fit before inverse_transform_target.")
        denom = self._target_max - self._target_min
        if denom == 0:
            return np.full_like(y, self._target_min)
        return y * denom + self._target_min

    def inverse_transform_uncertainty(self, sigma: np.ndarray) -> np.ndarray:
        """Scale uncertainty from normalized to original scale.

        Uncertainty (standard deviation) is a *spread* measure that scales
        linearly with the target range: σ_orig = σ_norm × (max − min).
        """
        if self._target_min is None:
            raise RuntimeError("Preprocessor must be fit before inverse_transform_uncertainty.")
        denom = self._target_max - self._target_min
        if denom == 0:
            return sigma.copy()
        return sigma * denom

    @property
    def target_range(self) -> float:
        """Return (target_max - target_min) from the fitted training data."""
        if self._target_min is None:
            raise RuntimeError("Preprocessor must be fit first.")
        return self._target_max - self._target_min





    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        denom = self._feature_max - self._feature_min
        denom[denom == 0] = 1.0


        return (X - self._feature_min) / denom

    def _scale_target(self, y: np.ndarray) -> np.ndarray:
        denom = self._target_max - self._target_min
        if denom == 0:
            return np.zeros_like(y)
        return (y - self._target_min) / denom
