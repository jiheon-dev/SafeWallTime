"""Comprehensive unit tests for all core UARP components.

Covers:
    - SWFLoader (parsing, filtering, sentinel handling)
    - UARPPreprocessor (MinMax scaling, NaN handling, inverse transforms)
    - AdaptiveMargin (margin formula)
    - FCFSEvaluator (discrete-event FCFS)
    - compute_prediction_metrics (prediction-level metrics)
    - JobTrace (temporal split, boundary handling)
    - Pipeline integration (end-to-end on synthetic data)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewalltime.data.base import JobTrace, REQUIRED_COLUMNS
from safewalltime.data.preprocessor import UARPPreprocessor
from safewalltime.evaluation.metrics import (
    FCFSEvaluator,
    EASYBackfillEvaluator,
    compute_prediction_metrics,
)
from safewalltime.margin.strategies import AdaptiveMargin
from safewalltime.models.base import PredictionResult






def _make_trace(jobs: list[dict]) -> JobTrace:
    """Build a JobTrace from a list of job dicts."""
    cols = ["job_id", "submit_time", "queue", "num_procs", "walltime", "memory", "runtime"]
    for j in jobs:
        for c in cols:
            j.setdefault(c, 0)
    df = pd.DataFrame(jobs, columns=cols)
    return JobTrace(df=df, format_name="test")


def _make_df(jobs: list[dict]) -> pd.DataFrame:
    """Build a trace DataFrame from a list of job dicts."""
    cols = ["job_id", "submit_time", "queue", "num_procs", "walltime", "memory", "runtime"]
    for j in jobs:
        for c in cols:
            j.setdefault(c, 0)
    return pd.DataFrame(jobs, columns=cols)






class TestSWFLoader:
    def _write_swf(self, lines: list[str], tmpdir: Path) -> str:
        """Write SWF lines to a temp file."""
        path = tmpdir / "test.swf"
        path.write_text("\n".join(lines) + "\n")
        return str(path)

    def test_basic_parsing(self, tmp_path):
        from safewalltime.data.swf import SWFLoader

        lines = [
            "; comment line",
            "1  0  10  100  4  50  1024  8  3600  2048  1  101  201  -1  0  0  -1  0",
            "2  100  5  200  2  30  512  4  7200  1024  1  102  202  -1  0  0  -1  0",
        ]
        loader = SWFLoader(min_runtime=0)
        trace = loader.load(self._write_swf(lines, tmp_path))
        assert trace.n_jobs == 2
        assert "runtime" in trace.df.columns
        assert "walltime" in trace.df.columns

    def test_filters_status_zero(self, tmp_path):
        from safewalltime.data.swf import SWFLoader
        lines = [
            "1  0  10  100  4  50  1024  8  3600  2048  1  101  201  -1  0  0  -1  0",
            "2  100  5  200  2  30  512  4  7200  1024  0  102  202  -1  0  0  -1  0",
        ]
        loader = SWFLoader(min_runtime=0, filter_status=True)
        trace = loader.load(self._write_swf(lines, tmp_path))
        assert trace.n_jobs == 1

    def test_walltime_sentinel_becomes_nan(self, tmp_path):
        """walltime=-1 should be stored as NaN, not clipped to 0."""
        from safewalltime.data.swf import SWFLoader
        lines = [

            "1  0  10  100  4  50  1024  8  -1  2048  1  101  201  -1  0  0  -1  0",
            "2  100  5  200  2  30  512  4  3600  1024  1  102  202  -1  0  0  -1  0",
        ]
        loader = SWFLoader(min_runtime=0)
        trace = loader.load(self._write_swf(lines, tmp_path))
        wt = trace.df["walltime"].values
        assert np.isnan(wt[0]), f"Expected NaN for walltime=-1, got {wt[0]}"
        assert wt[1] == 3600.0

    def test_min_runtime_filter(self, tmp_path):
        from safewalltime.data.swf import SWFLoader
        lines = [
            "1  0  10  5  4  50  1024  8  3600  2048  1  101  201  -1  0  0  -1  0",
            "2  100  5  200  2  30  512  4  7200  1024  1  102  202  -1  0  0  -1  0",
        ]
        loader = SWFLoader(min_runtime=10)
        trace = loader.load(self._write_swf(lines, tmp_path))
        assert trace.n_jobs == 1
        assert trace.df["runtime"].iloc[0] == 200

    def test_negative_runtime_filtered(self, tmp_path):
        from safewalltime.data.swf import SWFLoader
        lines = [
            "1  0  10  -1  4  50  1024  8  3600  2048  1  101  201  -1  0  0  -1  0",
            "2  100  5  200  2  30  512  4  7200  1024  1  102  202  -1  0  0  -1  0",
        ]
        loader = SWFLoader(min_runtime=0)
        trace = loader.load(self._write_swf(lines, tmp_path))
        assert trace.n_jobs == 1






class TestJobTrace:
    def test_requires_columns(self):
        df = pd.DataFrame({"job_id": [1], "submit_time": [0]})
        with pytest.raises(ValueError, match="missing required columns"):
            JobTrace(df=df)

    def test_temporal_split_ratio(self):
        trace = _make_trace([
            {"job_id": i, "submit_time": i * 10, "runtime": 50}
            for i in range(100)
        ])
        train, test = trace.temporal_split(0.7)

        assert train.n_jobs >= 70
        assert train.n_jobs + test.n_jobs == 100

    def test_temporal_split_boundary_same_submit_time(self):
        """Jobs with identical submit_time at boundary stay in train."""

        jobs = [{"job_id": i, "submit_time": i, "runtime": 10} for i in range(7)]
        jobs += [{"job_id": i, "submit_time": 7, "runtime": 10} for i in range(7, 10)]
        trace = _make_trace(jobs)
        train, test = trace.temporal_split(0.7)

        assert train.n_jobs == 10
        assert test.n_jobs == 0

    def test_temporal_split_preserves_order(self):
        trace = _make_trace([
            {"job_id": 3, "submit_time": 30, "runtime": 10},
            {"job_id": 1, "submit_time": 10, "runtime": 10},
            {"job_id": 2, "submit_time": 20, "runtime": 10},
        ])
        train, test = trace.temporal_split(0.7)

        assert list(train.df["submit_time"]) == sorted(train.df["submit_time"])






class TestPreprocessor:
    def test_fit_transform_scales_to_01(self):
        trace = _make_trace([
            {"job_id": i, "submit_time": i * 10, "num_procs": 10,
             "walltime": 100, "memory": 512, "queue": 0, "runtime": i * 100 + 100}
            for i in range(10)
        ])
        prep = UARPPreprocessor()
        X, y = prep.fit_transform(trace)

        assert X.shape[0] == 10
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_inverse_transform_roundtrip(self):
        trace = _make_trace([
            {"job_id": i, "submit_time": i, "runtime": (i + 1) * 100}
            for i in range(20)
        ])
        prep = UARPPreprocessor()
        _, y = prep.fit_transform(trace)
        y_orig = prep.inverse_transform_target(y)
        expected = trace.df["runtime"].values.astype(np.float64)
        np.testing.assert_allclose(y_orig, expected, rtol=1e-10)

    def test_inverse_transform_uncertainty(self):
        trace = _make_trace([
            {"job_id": i, "submit_time": i, "runtime": (i + 1) * 100}
            for i in range(20)
        ])
        prep = UARPPreprocessor()
        prep.fit_transform(trace)

        sigma_norm = np.array([0.1, 0.2, 0.5])
        sigma_orig = prep.inverse_transform_uncertainty(sigma_norm)
        denom = prep._target_max - prep._target_min
        np.testing.assert_allclose(sigma_orig, sigma_norm * denom)

    def test_nan_walltime_handled(self):
        """Preprocessor should handle NaN walltime without errors."""
        df = pd.DataFrame({
            "job_id": [1, 2, 3],
            "submit_time": [0, 10, 20],
            "queue": [0, 0, 0],
            "num_procs": [4, 8, 4],
            "walltime": [100, np.nan, 200],
            "memory": [512, 1024, 512],
            "runtime": [50, 80, 150],
        })
        trace = JobTrace(df=df)
        prep = UARPPreprocessor()
        X, y = prep.fit_transform(trace)



        wt_idx = prep._feature_names.index("walltime")
        assert np.isnan(X[1, wt_idx])

        assert not np.isnan(X[0, wt_idx])

    def test_constant_target_handled(self):
        """If all runtimes are the same, scaling should not crash."""
        trace = _make_trace([
            {"job_id": i, "submit_time": i, "runtime": 100} for i in range(5)
        ])
        prep = UARPPreprocessor()
        X, y = prep.fit_transform(trace)
        assert np.all(y == 0.0)
        y_orig = prep.inverse_transform_target(y)
        np.testing.assert_allclose(y_orig, 100.0)






class TestMargin:
    def test_adaptive_margin_formula(self):
        """margin = max(α·Q₀.₉₉, β·σ), t_safe = Q₀.₉₉ + margin."""
        alpha, beta = 0.2, 0.5
        q99 = np.array([100.0, 200.0, 50.0])
        sigma = np.array([10.0, 500.0, 5.0])

        pred = PredictionResult(
            point=q99.copy(),
            quantiles={0.99: q99.copy()},
            uncertainty=sigma.copy(),
        )
        margin_strat = AdaptiveMargin(alpha=alpha, beta=beta)
        est = margin_strat.compute(pred)

        expected_margin = np.maximum(alpha * q99, beta * sigma)
        expected_t_safe = q99 + expected_margin
        np.testing.assert_allclose(est.t_safe, expected_t_safe)
        np.testing.assert_allclose(est.margin, expected_margin)

    def test_adaptive_margin_no_uncertainty(self):
        """Falls back to conservative-only when uncertainty is None."""
        pred = PredictionResult(
            point=np.array([100.0]),
            quantiles={0.99: np.array([100.0])},
            uncertainty=None,
        )
        est = AdaptiveMargin(alpha=0.2).compute(pred)
        np.testing.assert_allclose(est.t_safe, [120.0])

class TestPredictionMetrics:
    def test_perfect_prediction(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])
        m = compute_prediction_metrics(y_true, y_pred)
        assert m.coverage_rate == 100.0
        assert m.underestimation_rate == 0.0
        assert m.mae == 0.0

    def test_all_underpredicted(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([50, 100, 150])
        m = compute_prediction_metrics(y_true, y_pred)
        assert m.coverage_rate == 0.0
        assert m.underestimation_rate == 100.0

    def test_waste_computation(self):
        y_true = np.array([100.0])
        y_pred = np.array([150.0])
        m = compute_prediction_metrics(y_true, y_pred)
        assert m.total_waste == 50.0
        assert m.mean_waste == 50.0

    def test_empty_input(self):
        m = compute_prediction_metrics(np.array([]), np.array([]))
        assert m.coverage_rate == 0.0
        assert m.mae == 0.0






class TestFCFSEvaluator:
    def test_simple_fcfs(self):
        df = _make_df([
            {"job_id": 1, "submit_time": 0, "num_procs": 10, "walltime": 100, "runtime": 50},
            {"job_id": 2, "submit_time": 10, "num_procs": 10, "walltime": 100, "runtime": 50},
        ])
        ev = FCFSEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=df["walltime"].values)
        assert m.nb_jobs_success == 2
        assert m.nb_jobs_killed == 0

    def test_fcfs_head_of_line_blocking(self):
        """In FCFS, a large job at head blocks smaller jobs behind it."""
        df = _make_df([
            {"job_id": 1, "submit_time": 0, "num_procs": 90, "walltime": 200, "runtime": 200},

            {"job_id": 2, "submit_time": 5, "num_procs": 90, "walltime": 100, "runtime": 50},
            {"job_id": 3, "submit_time": 10, "num_procs": 10, "walltime": 50, "runtime": 30},
        ])
        ev = FCFSEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=df["walltime"].values)
        assert m.nb_jobs_success == 3

        assert m.mean_wait_time > 0

    def test_fcfs_kill(self):
        df = _make_df([
            {"job_id": 1, "submit_time": 0, "num_procs": 10, "walltime": 100, "runtime": 200},
        ])
        ev = FCFSEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=np.array([50.0]))
        assert m.nb_jobs_killed == 1

    def test_fcfs_rejection(self):
        df = _make_df([
            {"job_id": 1, "submit_time": 0, "num_procs": 200, "walltime": 100, "runtime": 50},
        ])
        ev = FCFSEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=df["walltime"].values)
        assert m.nb_jobs_rejected == 1






class TestMarginScale:
    def test_margin_uses_original_q99_not_shifted(self):
        """Verify margin = max(α·Q₀.₉₉, β·σ) on ORIGINAL scale.

        Previously, margin was computed in normalized space, giving
        α·(Q₀.₉₉ - min_runtime) instead of α·Q₀.₉₉.
        """
        alpha, beta = 0.2, 0.5


        q99_orig = np.array([1000.0])
        sigma_orig = np.array([10.0])

        pred = PredictionResult(
            point=q99_orig.copy(),
            quantiles={0.99: q99_orig.copy()},
            uncertainty=sigma_orig.copy(),
        )
        est = AdaptiveMargin(alpha=alpha, beta=beta).compute(pred)


        expected_margin = 200.0
        expected_t_safe = 1000.0 + 200.0
        assert abs(est.margin[0] - expected_margin) < 1e-6
        assert abs(est.t_safe[0] - expected_t_safe) < 1e-6






if __name__ == "__main__":
    pytest.main([__file__, "-v"])
