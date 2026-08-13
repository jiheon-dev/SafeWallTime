"""SafeWallTime -- Uncertainty-Aware Runtime Prediction framework for HPC scheduling.

Public API
----------
All abstract base classes and concrete implementations are re-exported here
so that users can ``from safewalltime import ...`` without knowing the internal
package layout.

Quick start::

    from safewalltime import (
        UARPPipeline, SWFLoader, UARPPreprocessor,
        MultiQuantilePredictor, AdaptiveMargin, EASYBackfillEvaluator,
        StratifiedCQR,
    )

    pipeline = UARPPipeline(
        loader=SWFLoader(min_runtime=10),
        preprocessor=UARPPreprocessor(),
        predictor=MultiQuantilePredictor(quantiles=[0.50, 0.99]),
        margin=AdaptiveMargin(alpha=0.2, beta=0.5),
        evaluator=EASYBackfillEvaluator(num_nodes=100),
        # Optional: calibrate Q0.99 on a held-out temporal split.
        calibrator=StratifiedCQR(target_coverage=0.99, n_strata=10),
    )

    result = pipeline.run("path/to/trace.swf")
"""


from .data.base import JobTrace, Preprocessor, TraceLoader
from .models.base import PredictionResult, RuntimePredictor
from .margin.strategies import MarginStrategy, SchedulerEstimate
from .evaluation.metrics import (
    SchedulingEvaluator,
    SchedulingMetrics,
    PredictionMetrics,
    compute_prediction_metrics,
)


from .data.swf import SWFLoader
from .data.preprocessor import UARPPreprocessor
from .models.quantile import MultiQuantilePredictor
from .calibration.base import PredictionCalibrator
from .calibration.scqr import StratifiedCQR
from .margin.strategies import AdaptiveMargin
from .evaluation.metrics import FCFSEvaluator, EASYBackfillEvaluator
from .pipeline import UARPPipeline, PipelineResult

__version__ = "0.1.0"

__all__ = [

    "JobTrace",
    "Preprocessor",
    "TraceLoader",
    "PredictionResult",
    "RuntimePredictor",
    "MarginStrategy",
    "SchedulerEstimate",
    "SchedulingEvaluator",
    "SchedulingMetrics",
    "PredictionMetrics",
    "compute_prediction_metrics",

    "SWFLoader",
    "UARPPreprocessor",
    "MultiQuantilePredictor",
    "PredictionCalibrator",
    "StratifiedCQR",
    "AdaptiveMargin",
    "FCFSEvaluator",
    "EASYBackfillEvaluator",
    "UARPPipeline",
    "PipelineResult",
]
