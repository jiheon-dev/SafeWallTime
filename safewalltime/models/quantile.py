from __future__ import annotations
"""UARP multi-quantile regression with residual uncertainty modeling.

Implements Algorithm 1 from the UARP paper:
    1. Train Q_0.50 and Q_0.99 LightGBM quantile models
    2. Train residual uncertainty model on squared prediction errors from Q_0.50
    3. Predict: return Q_0.99 as point prediction, sigma_residual as uncertainty

This implementation is designed to be extensible:
    - Override `_create_base_model()` to swap LightGBM for another learner
    - Pass custom `quantiles` to change which percentiles are estimated
    - Pass custom `model_params` to tune the underlying model
"""

from typing import Any, Optional

import numpy as np

from .base import PredictionResult, RuntimePredictor


class MultiQuantilePredictor(RuntimePredictor):
    """Multi-quantile regression with residual uncertainty modeling.

    Args:
        quantiles: List of quantile levels to estimate. Default: [0.50, 0.99].
                   Must include ``residual_quantile`` (default 0.50).
        point_quantile: Which quantile to use as the point prediction. Default: 0.99.
        residual_quantile: Which quantile to compute residuals from (paper Eq. 1).
                           Default: 0.50 (median), matching the paper's definition:
                           sigma_residual = sqrt(E[(y - Q_0.50)^2 | x]).
        model_params: Dict of parameters passed to the underlying model.
        sample_weight_fn: Optional callable (y,) -> weights for training samples.
                          Use this to implement long-tail rebalancing.
    """

    def __init__(
        self,
        quantiles: Optional[list[float]] = None,
        point_quantile: float = 0.99,
        residual_quantile: float = 0.50,
        model_params: Optional[dict[str, Any]] = None,
        sample_weight_fn: Optional[callable] = None,
    ):
        self.quantiles = quantiles or [0.50, 0.99]
        self.point_quantile = point_quantile
        self.residual_quantile = residual_quantile
        self.model_params = model_params or {}
        self.sample_weight_fn = sample_weight_fn

        if self.point_quantile not in self.quantiles:
            raise ValueError(
                f"point_quantile={self.point_quantile} must be in quantiles={self.quantiles}"
            )
        if self.residual_quantile not in self.quantiles:
            raise ValueError(
                f"residual_quantile={self.residual_quantile} must be in quantiles={self.quantiles}. "
                f"The paper requires Q_0.50 for residual computation (Algorithm 1, line 6)."
            )

        self._quantile_models: dict[float, Any] = {}
        self._residual_model: Optional[Any] = None

    def _create_base_model(self, quantile: Optional[float] = None) -> Any:
        """Create a base model instance. Override to use a different learner.

        Args:
            quantile: If provided, create a quantile regression model at this level.
                      If None, create a standard regression model (for residual model).

        Returns:
            A model with fit(X, y, sample_weight=...) and predict(X) interface.
        """
        import lightgbm as lgb

        default_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 50,
            "verbose": -1,
            "n_jobs": -1,
        }
        params = {**default_params, **self.model_params}

        if quantile is not None:
            params["objective"] = "quantile"
            params["alpha"] = quantile
        else:
            params["objective"] = "regression"

        return lgb.LGBMRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiQuantilePredictor":
        """Train quantile models and residual uncertainty model (Algorithm 1, lines 1-7)."""
        sample_weight = None
        if self.sample_weight_fn is not None:
            sample_weight = self.sample_weight_fn(y)


        for tau in self.quantiles:
            model = self._create_base_model(quantile=tau)
            model.fit(X, y, sample_weight=sample_weight)
            self._quantile_models[tau] = model



        q_central = self._quantile_models[self.residual_quantile].predict(X)
        squared_residuals = (y - q_central) ** 2


        self._residual_model = self._create_base_model(quantile=None)
        self._residual_model.fit(X, squared_residuals, sample_weight=sample_weight)

        return self

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate predictions with uncertainty (Algorithm 1, lines 9-21)."""
        if not self._quantile_models:
            raise RuntimeError("Model must be fit before predict.")


        quantile_preds = {}
        for tau, model in self._quantile_models.items():
            quantile_preds[tau] = model.predict(X)


        sigma_residual = np.sqrt(
            np.maximum(self._residual_model.predict(X), 0)
        )


        point = quantile_preds[self.point_quantile].copy()

        return PredictionResult(
            point=point,
            quantiles=quantile_preds,
            uncertainty=sigma_residual,
            metadata={
                "point_quantile": self.point_quantile,
                "residual_quantile": self.residual_quantile,
            },
        )
