from __future__ import annotations
"""Abstract base classes for runtime prediction models.

To implement a custom predictor, subclass ``RuntimePredictor`` and return
a ``PredictionResult`` from ``predict()``.

Example::

    class MyPredictor(RuntimePredictor):
        def fit(self, X, y):
            ...
            return self

        def predict(self, X):
            return PredictionResult(
                point=my_predictions,
                quantiles={0.99: my_upper_bounds},
                uncertainty=my_sigma,
            )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class PredictionResult:
    """Container for model predictions.

    Attributes:
        point: Point predictions (typically Q_0.99).
        quantiles: Mapping from quantile level to predicted values.
        uncertainty: Per-sample uncertainty estimate (sigma_residual).
        metadata: Arbitrary extra info (model name, quantile used, etc.).
    """

    point: np.ndarray
    quantiles: dict[float, np.ndarray] = field(default_factory=dict)
    uncertainty: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

class RuntimePredictor(ABC):
    """Abstract base class for runtime prediction models.

    Any model that produces runtime predictions with optional uncertainty
    should subclass this.  The pipeline calls ``fit`` then ``predict``.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RuntimePredictor":
        """Train the model.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target runtimes (n_samples,).

        Returns:
            self (for chaining).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate predictions with optional uncertainty.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            PredictionResult containing point predictions and uncertainty.
        """
        ...
