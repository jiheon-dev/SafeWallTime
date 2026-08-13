"""Prediction calibration methods.

Calibration is kept separate from the predictor and margin stages.  This
lets a fitted UARP model be evaluated either as the original UARP baseline or
with a post-hoc calibration method such as S-CQR.
"""

from .base import PredictionCalibrator
from .scqr import StratifiedCQR

__all__ = ["PredictionCalibrator", "StratifiedCQR"]
