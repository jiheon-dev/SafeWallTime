"""Basic UARP pipeline example.

Run the full UARP pipeline on an SWF trace file:
    python examples/run_uarp.py path/to/trace.swf

Requires: pip install -e ".[lightgbm]"
"""

import sys

from safewalltime import (
    UARPPipeline,
    SWFLoader,
    UARPPreprocessor,
    MultiQuantilePredictor,
    AdaptiveMargin,
    EASYBackfillEvaluator,
)


def main(trace_path: str) -> None:
    pipeline = UARPPipeline(
        loader=SWFLoader(min_runtime=10),
        preprocessor=UARPPreprocessor(),
        predictor=MultiQuantilePredictor(quantiles=[0.50, 0.99]),
        margin=AdaptiveMargin(alpha=0.2, beta=0.5),
        evaluator=EASYBackfillEvaluator(num_nodes=100),
    )

    result = pipeline.run(trace_path)

    print(f"Dataset: {trace_path}")
    print(f"Total jobs: {result.trace.n_jobs}")
    print(f"Train / Test: {result.train_trace.n_jobs} / {result.test_trace.n_jobs}")
    print()
    print("=== Prediction Metrics ===")
    print(f"  Coverage rate:       {result.pred_metrics.coverage_rate:.2f}%")
    print(f"  Underestimation:     {result.pred_metrics.underestimation_rate:.2f}%")
    print(f"  MAE:                 {result.pred_metrics.mae:.1f}s")
    print(f"  Total waste:         {result.pred_metrics.total_waste / 3600:.1f}h")
    print(f"  Mean waste/job:      {result.pred_metrics.mean_waste:.1f}s")
    print()

    if result.sched_metrics:
        print("=== Scheduling Metrics (EASY Backfill) ===")
        print(f"  Utilization:    {result.sched_metrics.utilization:.2f}%")
        print(f"  Mean slowdown:  {result.sched_metrics.mean_slowdown:.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/run_uarp.py <path/to/trace.swf>")
        sys.exit(1)
    main(sys.argv[1])
