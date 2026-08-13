from __future__ import annotations
"""Abstract base classes for data loading and preprocessing.

Extension points:

* **TraceLoader** -- read job traces from any format (SWF, CSV, ...).
* **Preprocessor** -- feature extraction, scaling, and inverse transform.
* **JobTrace** -- wrapper around a pandas DataFrame with a
  canonical column contract and a temporal split helper.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd




REQUIRED_COLUMNS = frozenset({
    "job_id",
    "submit_time",
    "runtime",
})

OPTIONAL_COLUMNS = frozenset({
    "walltime",
    "num_procs",
    "queue",
    "memory",
    "user_id",
    "group_id",
    "status",
})




@dataclass
class JobTrace:
    """Immutable wrapper around a job-trace DataFrame.

    The DataFrame **must** contain at least the columns listed in
    ``REQUIRED_COLUMNS``.  Extra columns are allowed and preserved.

    Attributes:
        df: The underlying DataFrame (one row per job, sorted by submit_time).
        format_name: Human-readable name of the source format (e.g. "SWF").
        metadata: Arbitrary key-value info (dataset name, path, etc.).
    """

    df: pd.DataFrame
    format_name: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(
                f"JobTrace DataFrame is missing required columns: {missing}"
            )

    @property
    def n_jobs(self) -> int:
        return len(self.df)

    def temporal_split(self, train_ratio: float = 0.7) -> tuple["JobTrace", "JobTrace"]:
        """Split into train / test by submission order (preserves temporal ordering).

        Jobs with identical ``submit_time`` at the split boundary are kept
        together in the **training** set to avoid leaking concurrent-batch
        context into the test set.

        Args:
            train_ratio: Fraction of jobs assigned to the training set.

        Returns:
            (train_trace, test_trace)
        """
        df = self.df.sort_values("submit_time").reset_index(drop=True)
        split_idx = int(len(df) * train_ratio)


        if split_idx < len(df):
            boundary_time = df["submit_time"].iloc[split_idx]

            while split_idx < len(df) and df["submit_time"].iloc[split_idx] == boundary_time:
                split_idx += 1

        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        return (
            JobTrace(df=train_df, format_name=self.format_name, metadata=self.metadata),
            JobTrace(df=test_df, format_name=self.format_name, metadata=self.metadata),
        )




class TraceLoader(ABC):
    """Abstract base class for loading job traces from files.

    Subclass and implement ``load()`` to support new formats.

    Example::

        class CSVLoader(TraceLoader):
            def load(self, path):
                df = pd.read_csv(path)
                df = df.rename(columns={"run_time": "runtime", ...})
                return self._validate(df, format="CSV")
    """

    def __init__(self, min_runtime: float = 0.0) -> None:
        self.min_runtime = min_runtime

    @abstractmethod
    def load(self, path: str) -> JobTrace:
        """Load a trace file and return a ``JobTrace``.

        Args:
            path: Path to the trace file.

        Returns:
            A validated JobTrace.
        """
        ...

    def _validate(
        self,
        df: pd.DataFrame,
        format_name: str = "unknown",
        metadata: Optional[dict[str, Any]] = None,
    ) -> JobTrace:
        """Common post-load validation and filtering.

        * Drops rows where ``runtime <= 0`` or ``runtime < min_runtime``.
        * Sorts by ``submit_time``.
        * Resets the index.
        """
        df = df[df["runtime"] > 0].copy()
        if self.min_runtime > 0:
            df = df[df["runtime"] >= self.min_runtime]
        df = df.sort_values("submit_time").reset_index(drop=True)
        return JobTrace(df=df, format_name=format_name, metadata=metadata or {})




class Preprocessor(ABC):
    """Abstract base class for feature extraction and scaling.

    A preprocessor is fit on the **training** trace, then used to transform
    both training and test traces into ``(X, y)`` arrays.  It must also
    support inverse-transforming the target so that evaluation can happen
    in the original scale.
    """

    @abstractmethod
    def fit_transform(self, trace: JobTrace) -> tuple[np.ndarray, np.ndarray]:
        """Fit on a trace and return ``(X, y)``.

        Args:
            trace: Training JobTrace.

        Returns:
            (X, y) where X is (n_samples, n_features) and y is (n_samples,).
        """
        ...

    @abstractmethod
    def transform(self, trace: JobTrace) -> tuple[np.ndarray, np.ndarray]:
        """Transform a trace using previously fitted parameters.

        Args:
            trace: Test (or validation) JobTrace.

        Returns:
            (X, y) in the same format as ``fit_transform``.
        """
        ...

    @abstractmethod
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Map scaled target values back to original scale (seconds).

        Args:
            y: Scaled target array.

        Returns:
            Target array in original units.
        """
        ...

    @abstractmethod
    def inverse_transform_uncertainty(self, sigma: np.ndarray) -> np.ndarray:
        """Map scaled uncertainty (std dev) back to original scale.

        Uncertainty is a *spread* measure that scales with the target range,
        not shifted by the minimum.  For MinMax scaling:
        ``σ_orig = σ_norm × (max − min)``.

        Args:
            sigma: Scaled uncertainty array.

        Returns:
            Uncertainty array in original units.
        """
        ...
