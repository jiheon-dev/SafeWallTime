from __future__ import annotations
"""UARP Pipeline: end-to-end runtime prediction for HPC scheduling.

The pipeline connects data loading, preprocessing, prediction, margin
calculation, and evaluation into a single reproducible workflow.

Each component is pluggable — swap any part by passing a different
implementation of the corresponding abstract base class.

Usage:
    from safewalltime.pipeline import UARPPipeline
    from safewalltime.data import SWFLoader, UARPPreprocessor
    from safewalltime.models import MultiQuantilePredictor
    from safewalltime.margin import AdaptiveMargin
    from safewalltime.evaluation import EASYBackfillEvaluator

    pipeline = UARPPipeline(
        loader=SWFLoader(),
        preprocessor=UARPPreprocessor(),
        predictor=MultiQuantilePredictor(),
        margin=AdaptiveMargin(),
        evaluator=EASYBackfillEvaluator(num_nodes=100),
    )

    results = pipeline.run("path/to/trace.swf")
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .data.base import JobTrace, Preprocessor, TraceLoader
from .calibration.base import PredictionCalibrator
from .evaluation.metrics import (
    PredictionMetrics,
    SchedulingEvaluator,
    SchedulingMetrics,
    compute_prediction_metrics,
)
from .margin.strategies import MarginStrategy, SchedulerEstimate
from .models.base import PredictionResult, RuntimePredictor


@dataclass
class PipelineResult:
    """Container for all outputs from a pipeline run.

    Attributes:
        trace: The loaded job trace.
        train_trace: Training split of the trace.
        test_trace: Test split of the trace.
        prediction: Test-set prediction in original units, optionally calibrated.
        estimate: Scheduler-ready estimates (with margins) on test set.
        pred_metrics: Prediction-level evaluation metrics.
        sched_metrics: Scheduling-level evaluation metrics (if evaluator provided).
        y_true: Actual runtimes (original scale) for test set.
        y_pred: Predicted safe runtimes (original scale) for test set.
        calibration_trace: Held-out calibration split, if a calibrator is used.
    """

    trace: JobTrace
    train_trace: JobTrace
    test_trace: JobTrace
    prediction: PredictionResult
    estimate: SchedulerEstimate
    pred_metrics: PredictionMetrics
    sched_metrics: Optional[SchedulingMetrics]
    y_true: np.ndarray
    y_pred: np.ndarray
    calibration_trace: Optional[JobTrace] = None


def inverse_transform_prediction(
    prediction: PredictionResult,
    preprocessor: Preprocessor,
) -> PredictionResult:
    """Inverse-transform a PredictionResult from normalized to original scale.

    This must be done BEFORE margin computation so that the adaptive margin
    formula ``margin = max(α·Q₀.₉₉, β·σ)`` operates on the original-scale
    values as intended by the paper.  Computing margin in normalized space
    would give ``α·(Q₀.₉₉ − min_runtime)`` instead of ``α·Q₀.₉₉``.
    """
    point_orig = preprocessor.inverse_transform_target(prediction.point)
    quantiles_orig = {
        q: preprocessor.inverse_transform_target(v)
        for q, v in prediction.quantiles.items()
    }
    uncertainty_orig = (
        preprocessor.inverse_transform_uncertainty(prediction.uncertainty)
        if prediction.uncertainty is not None
        else None
    )
    return PredictionResult(
        point=point_orig,
        quantiles=quantiles_orig,
        uncertainty=uncertainty_orig,
        metadata=prediction.metadata,
    )


class UARPPipeline:
    """End-to-end pipeline for uncertainty-aware runtime prediction.

    Args:
        loader: TraceLoader for reading job trace files.
        preprocessor: Preprocessor for feature extraction and scaling.
        predictor: RuntimePredictor model.
        margin: MarginStrategy for computing safety margins.
        evaluator: Optional SchedulingEvaluator for scheduling simulation.
        calibrator: Optional post-hoc calibrator, for example ``StratifiedCQR``.
                    It is fit on a held-out temporal split before test evaluation.
        train_ratio: Temporal train/test split ratio.
        calibration_ratio: Fraction of the training partition reserved for
                           calibration when ``calibrator`` is provided.
    """

    def __init__(
        self,
        loader: TraceLoader,
        preprocessor: Preprocessor,
        predictor: RuntimePredictor,
        margin: MarginStrategy,
        evaluator: Optional[SchedulingEvaluator] = None,
        calibrator: Optional[PredictionCalibrator] = None,
        train_ratio: float = 0.7,
        calibration_ratio: float = 0.15,
    ):
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1.")
        if calibrator is not None and not 0.0 < calibration_ratio < 1.0:
            raise ValueError("calibration_ratio must be between 0 and 1.")
        self.loader = loader
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.margin = margin
        self.calibrator = calibrator
        self.evaluator = evaluator
        self.train_ratio = train_ratio
        self.calibration_ratio = calibration_ratio

    def _run_core(
        self,
        train_trace: JobTrace,
        test_trace: JobTrace,
        calibration_trace: Optional[JobTrace] = None,
    ) -> tuple[PredictionResult, SchedulerEstimate, np.ndarray, np.ndarray]:
        """Shared logic for run() and run_from_trace().

        Returns:
            (prediction_orig, estimate, y_true_original, t_safe_original)
        """

        X_train, y_train = self.preprocessor.fit_transform(train_trace)
        X_test, y_test = self.preprocessor.transform(test_trace)


        self.predictor.fit(X_train, y_train)

        if calibration_trace is not None:
            if self.calibrator is None:
                raise RuntimeError("A calibration trace requires a calibrator.")
            X_calibration, y_calibration = self.preprocessor.transform(calibration_trace)
            calibration_prediction = inverse_transform_prediction(
                self.predictor.predict(X_calibration), self.preprocessor,
            )
            y_calibration_original = self.preprocessor.inverse_transform_target(
                y_calibration,
            )
            self.calibrator.fit(calibration_prediction, y_calibration_original)

        prediction_norm = self.predictor.predict(X_test)


        y_true_original = self.preprocessor.inverse_transform_target(y_test)

        prediction_orig = inverse_transform_prediction(
            prediction_norm, self.preprocessor,
        )

        if self.calibrator is not None:
            prediction_orig = self.calibrator.transform(prediction_orig)


        estimate = self.margin.compute(prediction_orig)
        t_safe_original = estimate.t_safe

        return prediction_orig, estimate, y_true_original, t_safe_original

    def _split_trace(
        self,
        trace: JobTrace,
    ) -> tuple[JobTrace, Optional[JobTrace], JobTrace]:
        """Create temporal train/(optional calibration)/test partitions."""
        train_trace, test_trace = trace.temporal_split(self.train_ratio)
        if self.calibrator is None:
            return train_trace, None, test_trace

        model_train, calibration_trace = train_trace.temporal_split(
            1.0 - self.calibration_ratio,
        )
        if model_train.n_jobs == 0 or calibration_trace.n_jobs == 0:
            raise ValueError(
                "The selected temporal split leaves no data for model training or "
                "calibration. Adjust train_ratio or calibration_ratio."
            )
        return model_train, calibration_trace, test_trace

    def _run_trace(self, trace: JobTrace) -> PipelineResult:
        """Run the pipeline from a loaded trace."""
        train_trace, calibration_trace, test_trace = self._split_trace(trace)
        prediction_orig, estimate, y_true_original, t_safe_original = self._run_core(
            train_trace, test_trace, calibration_trace,
        )

        pred_metrics = compute_prediction_metrics(y_true_original, t_safe_original)

        sched_metrics = None
        if self.evaluator is not None:
            sched_metrics = self.evaluator.evaluate(
                trace_df=test_trace.df,
                t_safe=t_safe_original,
            )

        return PipelineResult(
            trace=trace,
            train_trace=train_trace,
            calibration_trace=calibration_trace,
            test_trace=test_trace,
            prediction=prediction_orig,
            estimate=estimate,
            pred_metrics=pred_metrics,
            sched_metrics=sched_metrics,
            y_true=y_true_original,
            y_pred=t_safe_original,
        )

    def run(self, trace_path: str) -> PipelineResult:
        """Execute the full pipeline.

        Steps:
            1. Load trace
            2. Temporal train/test split
            3. Optionally split training data into model-training and calibration sets
            4. Fit preprocessor and predictor on model-training data
            5. Optionally fit the calibrator on held-out calibration data
            6. Predict on test set and inverse-transform to original scale
            7. Apply calibration and compute safety margins
            8. Evaluate prediction and optional scheduling metrics

        Args:
            trace_path: Path to the trace file.

        Returns:
            PipelineResult with all outputs.
        """
        return self._run_trace(self.loader.load(trace_path))

    def run_from_trace(self, trace: JobTrace) -> PipelineResult:
        """Execute pipeline from a pre-loaded trace (skip loading step)."""
        return self._run_trace(trace)
