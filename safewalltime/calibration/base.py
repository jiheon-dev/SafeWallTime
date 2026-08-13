"""Interfaces for post-hoc prediction calibration."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models.base import PredictionResult


class PredictionCalibrator(ABC):
    """Post-process a model prediction using held-out calibration data.

    A calibrator is fit after the runtime predictor and before the margin
    strategy.  Both the calibration targets and predictions must use the same
    units; :class:`~safewalltime.pipeline.UARPPipeline` supplies original-scale values.
    """

    @abstractmethod
    def fit(
        self,
        prediction: PredictionResult,
        y_true: np.ndarray,
    ) -> "PredictionCalibrator":
        """Fit calibration parameters from held-out predictions and targets."""
        ...

    @abstractmethod
    def transform(self, prediction: PredictionResult) -> PredictionResult:
        """Return a calibrated copy of ``prediction``."""
        ...

    def fit_transform(
        self,
        prediction: PredictionResult,
        y_true: np.ndarray,
    ) -> PredictionResult:
        """Fit the calibrator and transform the same prediction."""
        return self.fit(prediction, y_true).transform(prediction)
