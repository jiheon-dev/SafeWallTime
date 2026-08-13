"""Tests for the public S-CQR calibration API and pipeline integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from safewalltime import (
    AdaptiveMargin,
    JobTrace,
    PredictionResult,
    RuntimePredictor,
    StratifiedCQR,
    UARPPipeline,
    UARPPreprocessor,
)


def _prediction(q50: np.ndarray, q99: np.ndarray) -> PredictionResult:
    return PredictionResult(
        point=q99.copy(),
        quantiles={0.50: q50.copy(), 0.99: q99.copy()},
        uncertainty=np.full(len(q99), 0.1),
    )


class _FeatureQuantilePredictor(RuntimePredictor):
    """Small deterministic predictor used to exercise the pipeline contract."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FeatureQuantilePredictor":
        return self

    def predict(self, X: np.ndarray) -> PredictionResult:
        q50 = X[:, 0].astype(float)
        q99 = q50 + 0.1
        return _prediction(q50, q99)


class TestStratifiedCQR:
    def test_applies_distinct_finite_sample_corrections_per_stratum(self):
        q50 = np.concatenate((np.arange(10.0), np.arange(100.0, 110.0)))
        q99 = np.full(20, 10.0)
        y_true = np.concatenate((np.arange(10.0), np.arange(10.0, 20.0)))
        calibrator = StratifiedCQR(
            target_coverage=0.8,
            n_strata=2,
            min_samples_per_stratum=10,
        ).calibrate(q99, q50, y_true)


        assert calibrator.corrections == {0: -2.0, 1: 8.0}
        np.testing.assert_allclose(
            calibrator.predict(np.array([10.0, 10.0]), np.array([1.0, 101.0])),
            np.array([8.0, 18.0]),
        )

    def test_transform_returns_a_calibrated_copy(self):
        q50 = np.arange(10.0)
        q99 = q50 + 2.0
        prediction = _prediction(q50, q99)
        calibrator = StratifiedCQR(
            target_coverage=0.8,
            n_strata=1,
            min_samples_per_stratum=10,
        ).fit(prediction, q99 + 3.0)

        calibrated = calibrator.transform(prediction)

        np.testing.assert_allclose(prediction.quantiles[0.99], q99)
        np.testing.assert_allclose(calibrated.quantiles[0.99], q99 + 3.0)
        np.testing.assert_allclose(calibrated.point, q99 + 3.0)
        np.testing.assert_allclose(calibrated.uncertainty, prediction.uncertainty)
        assert calibrated.metadata["calibrator"] == "s-cqr"

    def test_merges_tied_percentile_boundaries_without_empty_strata(self):
        q50 = np.repeat([0.0, 1.0], 10)
        q99 = np.full(20, 10.0)
        calibrator = StratifiedCQR(
            target_coverage=0.8,
            n_strata=10,
            min_samples_per_stratum=10,
        ).calibrate(q99, q50, q99)

        assert calibrator.n_strata == 2
        assert [info["n_calibration"] for info in calibrator.get_strata_info()] == [10, 10]

    def test_requires_the_quantiles_used_by_scqr(self):
        prediction = PredictionResult(point=np.ones(10), quantiles={0.99: np.ones(10)})
        calibrator = StratifiedCQR(n_strata=1)

        with pytest.raises(ValueError, match="Q0.50"):
            calibrator.fit(prediction, np.ones(10))


class TestSCQRPipeline:
    def test_reserves_a_temporal_calibration_split_and_applies_scqr(self):
        trace = JobTrace(
            df=pd.DataFrame(
                {
                    "job_id": np.arange(100),
                    "submit_time": np.arange(100),
                    "runtime": np.arange(100.0, 200.0),
                }
            )
        )
        pipeline = UARPPipeline(
            loader=object(),
            preprocessor=UARPPreprocessor(feature_cols=["job_id"]),
            predictor=_FeatureQuantilePredictor(),
            margin=AdaptiveMargin(alpha=0.0, beta=0.0),
            calibrator=StratifiedCQR(n_strata=1, min_samples_per_stratum=10),
            calibration_ratio=0.2,
        )

        result = pipeline.run_from_trace(trace)




        assert result.train_trace.n_jobs == 57
        assert result.calibration_trace is not None
        assert result.calibration_trace.n_jobs == 14
        assert result.test_trace.n_jobs == 29
        assert result.prediction.metadata["calibrator"] == "s-cqr"
        np.testing.assert_allclose(result.y_pred, result.prediction.quantiles[0.99])
