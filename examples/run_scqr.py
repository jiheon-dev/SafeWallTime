"""Run UARP with S-CQR calibration on an SWF trace.

Usage:
    python examples/run_scqr.py path/to/trace.swf

Requires: pip install -e ".[lightgbm]"
"""

import sys

from safewalltime import (
    AdaptiveMargin,
    EASYBackfillEvaluator,
    MultiQuantilePredictor,
    SWFLoader,
    StratifiedCQR,
    UARPPipeline,
    UARPPreprocessor,
)


def main(trace_path: str) -> None:
    pipeline = UARPPipeline(
        loader=SWFLoader(min_runtime=10),
        preprocessor=UARPPreprocessor(),
        predictor=MultiQuantilePredictor(quantiles=[0.50, 0.99]),
        calibrator=StratifiedCQR(target_coverage=0.99, n_strata=10),
        margin=AdaptiveMargin(alpha=0.2, beta=0.5),
        evaluator=EASYBackfillEvaluator(num_nodes=100),
    )
    result = pipeline.run(trace_path)

    print(f"Dataset: {trace_path}")
    print(
        "Model train / calibration / test: "
        f"{result.train_trace.n_jobs} / "
        f"{result.calibration_trace.n_jobs if result.calibration_trace else 0} / "
        f"{result.test_trace.n_jobs}"
    )
    print(f"Coverage rate: {result.pred_metrics.coverage_rate:.2f}%")
    print(f"Mean waste/job: {result.pred_metrics.mean_waste:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/run_scqr.py <path/to/trace.swf>")
        sys.exit(1)
    main(sys.argv[1])
