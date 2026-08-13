from __future__ import annotations
"""Safety margin calculation strategies.

The margin module converts raw predictions into scheduler-ready runtime estimates.
Different strategies balance between job completion (avoiding kills) and
resource efficiency (avoiding overprediction waste).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..models.base import PredictionResult


@dataclass
class SchedulerEstimate:
    """Final runtime estimate ready for the scheduler.

    Attributes:
        t_safe: Conservative runtime estimate (Q_0.99 + margin).
        margin: Per-job safety margin that was added.
        prediction: The underlying PredictionResult for analysis.
    """

    t_safe: np.ndarray
    margin: np.ndarray
    prediction: PredictionResult

    @property
    def n_jobs(self) -> int:
        return len(self.t_safe)


class MarginStrategy(ABC):
    """Abstract base class for safety margin calculation.

    Subclass this to implement different margin strategies:
    - UARP adaptive margin (default)
    - Fixed multiplier (baseline comparison)
    - Queue-pressure-aware adaptive margin
    - etc.

    The contract:
        - compute(prediction) takes a PredictionResult and returns SchedulerEstimate
    """

    @abstractmethod
    def compute(self, prediction: PredictionResult) -> SchedulerEstimate:
        """Compute safety margin and produce scheduler-ready estimates.

        Args:
            prediction: Output from a RuntimePredictor.

        Returns:
            SchedulerEstimate with t_safe = point_prediction + margin.
        """
        ...


class AdaptiveMargin(MarginStrategy):
    """UARP adaptive safety margin (Algorithm 2).

    Margin(x) = max(alpha * Q_0.99(x), beta * sigma_residual(x))

    The first term (conservative) sets a minimum margin proportional to
    the predicted runtime tail, protecting against systematic bias.
    The second term (uncertainty) adapts to per-job prediction confidence,
    giving larger buffers to jobs the model is uncertain about.

    Args:
        alpha: Scaling factor for the conservative term (default: 0.2).
        beta: Scaling factor for the uncertainty term (default: 0.5).
        point_quantile: Which quantile key to use from prediction (default: 0.99).
    """

    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 0.5,
        point_quantile: float = 0.99,
    ):
        self.alpha = alpha
        self.beta = beta
        self.point_quantile = point_quantile

    def compute(self, prediction: PredictionResult) -> SchedulerEstimate:
        q_upper = prediction.quantiles.get(self.point_quantile, prediction.point)
        sigma = prediction.uncertainty

        if sigma is None:

            margin = self.alpha * q_upper
        else:

            m_conservative = self.alpha * q_upper
            m_uncertainty = self.beta * sigma
            margin = np.maximum(m_conservative, m_uncertainty)

        t_safe = q_upper + margin

        return SchedulerEstimate(
            t_safe=t_safe,
            margin=margin,
            prediction=prediction,
        )
