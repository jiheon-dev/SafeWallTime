"""Stratified conformalized quantile regression (S-CQR).

S-CQR calibrates UARP's upper runtime quantile separately for groups of jobs
with similar central-quantile predictions.  It is a lightweight post-hoc
method: train the UARP predictor as usual, fit this calibrator on a held-out
time-ordered calibration split, then apply it to future predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import PredictionCalibrator
from ..models.base import PredictionResult


class StratifiedCQR(PredictionCalibrator):
    """Calibrate an upper quantile separately across central-quantile strata.

    The calibration score in stratum ``g`` is ``y - Q_upper``.  S-CQR adds the
    finite-sample conformal quantile of those scores to future upper-quantile
    predictions in the same stratum.  Negative corrections can therefore make
    systematically over-conservative UARP predictions tighter.

    Args:
        target_coverage: Desired marginal coverage for the calibrated upper
            bound.  For UARP's Q0.99 output, ``0.99`` is a natural default.
        n_strata: Maximum number of strata formed from central-quantile
            percentiles.  Duplicate percentile boundaries are merged.
        min_samples_per_stratum: Minimum held-out calibration observations in
            each resulting stratum.
        lower_quantile: Central quantile used to assign a job to a stratum.
        upper_quantile: Upper quantile adjusted by conformal calibration.

    Example:
        >>> calibrator = StratifiedCQR(target_coverage=0.99, n_strata=10)
        >>> calibrator.fit(calibration_prediction, calibration_targets)
        >>> calibrated_prediction = calibrator.transform(test_prediction)

    ``calibrate(q_upper, q_central, y_true)`` and
    ``predict(q_upper, q_central)`` are also provided for array-based research
    workflows.
    """

    def __init__(
        self,
        target_coverage: float = 0.99,
        n_strata: int = 10,
        min_samples_per_stratum: int = 10,
        lower_quantile: float = 0.50,
        upper_quantile: float = 0.99,
    ) -> None:
        if not 0.0 < target_coverage < 1.0:
            raise ValueError("target_coverage must be between 0 and 1.")
        if n_strata < 1:
            raise ValueError("n_strata must be at least 1.")
        if min_samples_per_stratum < 1:
            raise ValueError("min_samples_per_stratum must be at least 1.")
        if lower_quantile >= upper_quantile:
            raise ValueError("lower_quantile must be smaller than upper_quantile.")

        self.target_coverage = target_coverage
        self.requested_n_strata = n_strata
        self.n_strata = n_strata
        self.min_samples_per_stratum = min_samples_per_stratum
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

        self._edges: np.ndarray | None = None
        self._corrections: dict[int, float] = {}
        self._sample_counts: dict[int, int] = {}

    @property
    def edges(self) -> np.ndarray | None:
        """Stratum boundaries after fitting, including ``-inf`` and ``inf``."""
        return None if self._edges is None else self._edges.copy()

    @property
    def corrections(self) -> dict[int, float]:
        """Per-stratum additive corrections after fitting."""
        return self._corrections.copy()

    def fit(self, prediction: PredictionResult, y_true: np.ndarray) -> "StratifiedCQR":
        """Fit S-CQR from a held-out :class:`PredictionResult`."""
        q_upper = self._quantile_from_prediction(prediction, self.upper_quantile)
        q_central = self._quantile_from_prediction(prediction, self.lower_quantile)
        return self.calibrate(q_upper, q_central, y_true)

    def calibrate(
        self,
        q_upper: np.ndarray,
        q_central: np.ndarray,
        y_true: np.ndarray,
    ) -> "StratifiedCQR":
        """Fit S-CQR from arrays of upper, central, and observed runtimes.

        This is the array-oriented equivalent of :meth:`fit` and is useful for
        experiment code that stores quantile predictions separately.
        """
        upper, central, target = self._validate_arrays(q_upper, q_central, y_true)
        if len(target) < self.min_samples_per_stratum:
            raise ValueError(
                "Not enough calibration samples: "
                f"need at least {self.min_samples_per_stratum}, got {len(target)}."
            )




        percentiles = np.linspace(0.0, 100.0, self.requested_n_strata + 1)
        interior_edges = np.unique(np.percentile(central, percentiles)[1:-1])



        interior_edges = interior_edges[
            (interior_edges > central.min()) & (interior_edges < central.max())
        ]
        edges = np.concatenate(([-np.inf], interior_edges, [np.inf]))
        n_strata = len(edges) - 1
        groups = np.digitize(central, edges[1:-1])

        corrections: dict[int, float] = {}
        sample_counts: dict[int, int] = {}
        for group in range(n_strata):
            mask = groups == group
            n_group = int(mask.sum())
            if n_group < self.min_samples_per_stratum:
                raise ValueError(
                    f"Stratum {group} has {n_group} calibration samples; "
                    f"need at least {self.min_samples_per_stratum}. "
                    "Reduce n_strata or provide more calibration data."
                )

            scores = target[mask] - upper[mask]
            corrections[group] = self._conformal_quantile(scores)
            sample_counts[group] = n_group

        self._edges = edges
        self._corrections = corrections
        self._sample_counts = sample_counts

        self.n_strata = n_strata
        return self

    def transform(self, prediction: PredictionResult) -> PredictionResult:
        """Return a copy with its upper quantile calibrated by S-CQR."""
        q_upper = self._quantile_from_prediction(prediction, self.upper_quantile)
        q_central = self._quantile_from_prediction(prediction, self.lower_quantile)
        calibrated_upper = self.predict(q_upper, q_central)

        quantiles = {
            level: np.asarray(values, dtype=float).copy()
            for level, values in prediction.quantiles.items()
        }
        quantiles[self.upper_quantile] = calibrated_upper
        metadata: dict[str, Any] = {
            **prediction.metadata,
            "calibrator": "s-cqr",
            "target_coverage": self.target_coverage,
            "n_strata": self.n_strata,
        }
        return PredictionResult(
            point=calibrated_upper.copy(),
            quantiles=quantiles,
            uncertainty=(
                None
                if prediction.uncertainty is None
                else np.asarray(prediction.uncertainty, dtype=float).copy()
            ),
            metadata=metadata,
        )

    def predict(self, q_upper: np.ndarray, q_central: np.ndarray) -> np.ndarray:
        """Apply fitted per-stratum corrections to upper-quantile arrays."""
        if self._edges is None:
            raise RuntimeError("S-CQR must be calibrated before predict.")
        upper, central, _ = self._validate_arrays(q_upper, q_central, None)
        groups = np.digitize(central, self._edges[1:-1])
        corrections = np.asarray(
            [self._corrections[int(group)] for group in groups], dtype=float
        )
        return upper + corrections

    def get_strata_info(self) -> list[dict[str, Any]]:
        """Return fitted boundaries, corrections, and sample counts by stratum."""
        if self._edges is None:
            raise RuntimeError("S-CQR must be calibrated before requesting strata info.")
        return [
            {
                "stratum": group,
                "q50_range": (
                    0.0 if self._edges[group] == -np.inf else self._edges[group],
                    self._edges[group + 1],
                ),
                "correction": self._corrections[group],
                "n_calibration": self._sample_counts[group],
            }
            for group in range(self.n_strata)
        ]

    def _conformal_quantile(self, scores: np.ndarray) -> float:
        """Return the finite-sample upper conformal quantile of ``scores``."""
        n = len(scores)

        rank = min(int(np.ceil((n + 1) * self.target_coverage)), n)
        return float(np.partition(scores, rank - 1)[rank - 1])

    @staticmethod
    def _quantile_from_prediction(
        prediction: PredictionResult,
        quantile: float,
    ) -> np.ndarray:
        try:
            values = prediction.quantiles[quantile]
        except KeyError as exc:
            available = sorted(prediction.quantiles)
            raise ValueError(
                f"PredictionResult must include Q{quantile:.2f}; "
                f"available quantiles: {available}."
            ) from exc
        return np.asarray(values, dtype=float)

    @staticmethod
    def _validate_arrays(
        q_upper: np.ndarray,
        q_central: np.ndarray,
        y_true: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        upper = np.asarray(q_upper, dtype=float).reshape(-1)
        central = np.asarray(q_central, dtype=float).reshape(-1)
        target = None if y_true is None else np.asarray(y_true, dtype=float).reshape(-1)

        lengths = [len(upper), len(central)]
        if target is not None:
            lengths.append(len(target))
        if len(set(lengths)) != 1:
            raise ValueError("q_upper, q_central, and y_true must have the same length.")
        if len(upper) == 0:
            raise ValueError("Calibration arrays must not be empty.")
        arrays = [upper, central] if target is None else [upper, central, target]
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError("Calibration arrays must contain only finite values.")
        return upper, central, target
