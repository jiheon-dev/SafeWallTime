"""Unit tests for EASY backfill simulator correctness."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safewalltime.evaluation.metrics import EASYBackfillEvaluator


def _make_df(jobs: list[dict]) -> pd.DataFrame:
    """Build a trace DataFrame from a list of job dicts."""
    cols = ["job_id", "submit_time", "queue", "num_procs", "walltime", "memory", "runtime"]
    for j in jobs:
        for c in cols:
            j.setdefault(c, 0)
    return pd.DataFrame(jobs, columns=cols)





class TestFCFS:
    def test_jobs_start_in_submission_order(self):
        """When resources are ample, jobs start immediately in FCFS order."""
        df = _make_df([
            {"job_id": 1, "submit_time": 0,   "num_procs": 10, "walltime": 100, "runtime": 50},
            {"job_id": 2, "submit_time": 10,  "num_procs": 10, "walltime": 100, "runtime": 50},
            {"job_id": 3, "submit_time": 20,  "num_procs": 10, "walltime": 100, "runtime": 50},
        ])
        ev = EASYBackfillEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=df["walltime"].values)

        assert m.nb_jobs_success == 3
        assert m.nb_jobs_rejected == 0
        assert m.nb_jobs_killed == 0

    def test_queued_when_resources_busy(self):
        """Job B waits until Job A frees resources."""
        df = _make_df([
            {"job_id": 1, "submit_time": 0,  "num_procs": 80, "walltime": 100, "runtime": 100},
            {"job_id": 2, "submit_time": 10, "num_procs": 80, "walltime": 100, "runtime": 50},
        ])
        ev = EASYBackfillEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=df["walltime"].values)

        assert m.nb_jobs_success == 2

        assert m.mean_wait_time > 0





class TestBackfill:
    def test_small_job_backfills_before_large(self):
        """A small later job starts before a large earlier-queued job
        if it finishes before shadow time."""
        df = _make_df([

            {"job_id": 1, "submit_time": 0,  "num_procs": 90, "walltime": 200, "runtime": 200},

            {"job_id": 2, "submit_time": 5,  "num_procs": 80, "walltime": 300, "runtime": 100},

            {"job_id": 3, "submit_time": 10, "num_procs": 10, "walltime": 50,  "runtime": 30},
        ])
        ev = EASYBackfillEvaluator(num_nodes=100, collect_timeseries=True)
        m = ev.evaluate(df, t_safe=df["walltime"].values)

        assert m.nb_jobs_success == 3
        assert m.nb_jobs_killed == 0

    def test_backfill_blocked_if_past_shadow(self):
        """Job C cannot backfill if its walltime extends past shadow."""
        df = _make_df([
            {"job_id": 1, "submit_time": 0,   "num_procs": 90,  "walltime": 100, "runtime": 100},
            {"job_id": 2, "submit_time": 5,   "num_procs": 100, "walltime": 200, "runtime": 100},

            {"job_id": 3, "submit_time": 10,  "num_procs": 10,  "walltime": 500, "runtime": 30},
        ])
        ev = EASYBackfillEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=df["walltime"].values)


        assert m.nb_jobs_success == 3

    def test_condition2_extra_nodes(self):
        """Condition 2: job uses excess nodes at shadow time.

        Setup: 100 nodes.
          A: 60 nodes, walltime=200 (occupies cluster)
          B: 90 nodes, arrives at t=5, cannot start (60+90 > 100)
             shadow = 200 (when A finishes), at shadow: 100 nodes free, B needs 90 → extra=10
          C: 10 nodes, walltime=500 (extends past shadow)
             C should start via condition 2: 10 <= extra(10)
        """
        df = _make_df([
            {"job_id": 1, "submit_time": 0,   "num_procs": 60,  "walltime": 200, "runtime": 200},
            {"job_id": 2, "submit_time": 5,   "num_procs": 90,  "walltime": 300, "runtime": 100},
            {"job_id": 3, "submit_time": 10,  "num_procs": 10,  "walltime": 500, "runtime": 400},
        ])
        ev = EASYBackfillEvaluator(num_nodes=100, collect_per_job=True)
        m = ev.evaluate(df, t_safe=df["walltime"].values)

        per_job = m.extra["per_job"]
        assert per_job["wait_time"][1] == 195.0
        assert per_job["wait_time"][2] == 0.0

    def test_condition2_counts_all_resources_freed_at_shadow_time(self):
        """A backfill job may use all capacity left by the head job at shadow."""
        df = _make_df([
            {"job_id": 1, "submit_time": 0, "num_procs": 15, "walltime": 100, "runtime": 100},
            {"job_id": 2, "submit_time": 0, "num_procs": 15, "walltime": 100, "runtime": 100},
            {"job_id": 3, "submit_time": 0, "num_procs": 15, "walltime": 100, "runtime": 100},
            {"job_id": 4, "submit_time": 0, "num_procs": 15, "walltime": 100, "runtime": 100},
            {"job_id": 5, "submit_time": 0, "num_procs": 20, "walltime": 100, "runtime": 100},
            {"job_id": 6, "submit_time": 1, "num_procs": 80, "walltime": 100, "runtime": 100},
            {"job_id": 7, "submit_time": 2, "num_procs": 20, "walltime": 200, "runtime": 150},
        ])
        ev = EASYBackfillEvaluator(num_nodes=100, collect_per_job=True)
        m = ev.evaluate(df, t_safe=df["walltime"].values)

        per_job = m.extra["per_job"]
        assert per_job["wait_time"][5] == 99.0
        assert per_job["wait_time"][6] == 0.0
        assert m.makespan == 200.0





class TestRejection:
    def test_oversized_job_rejected(self):
        """Job requesting more nodes than cluster is rejected."""
        df = _make_df([
            {"job_id": 1, "submit_time": 0, "num_procs": 50,  "walltime": 100, "runtime": 50},
            {"job_id": 2, "submit_time": 5, "num_procs": 200, "walltime": 100, "runtime": 50},
        ])
        ev = EASYBackfillEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=df["walltime"].values)

        assert m.nb_jobs_rejected == 1
        assert m.nb_jobs_success == 1





class TestKilling:
    def test_job_killed_when_runtime_exceeds_walltime(self):
        """Job is killed when actual runtime > walltime (t_safe)."""
        df = _make_df([
            {"job_id": 1, "submit_time": 0, "num_procs": 10, "walltime": 100, "runtime": 200},
        ])

        ev = EASYBackfillEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=np.array([50.0]))

        assert m.nb_jobs_killed == 1
        assert m.nb_jobs_success == 0

    def test_job_survives_when_walltime_sufficient(self):
        """Job completes when t_safe >= runtime."""
        df = _make_df([
            {"job_id": 1, "submit_time": 0, "num_procs": 10, "walltime": 100, "runtime": 50},
        ])
        ev = EASYBackfillEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=np.array([100.0]))

        assert m.nb_jobs_killed == 0
        assert m.nb_jobs_success == 1





class TestResources:
    def test_rejects_invalid_scheduler_inputs(self):
        df = _make_df([
            {"job_id": 1, "submit_time": 0, "num_procs": 10, "walltime": 100, "runtime": 50},
        ])

        with pytest.raises(ValueError, match="strictly positive"):
            EASYBackfillEvaluator(num_nodes=100).evaluate(df, np.array([0.0]))

        with pytest.raises(ValueError, match="every job"):
            EASYBackfillEvaluator(num_nodes=100).evaluate(df, np.array([]))

        with pytest.raises(ValueError, match="num_nodes"):
            EASYBackfillEvaluator(num_nodes=0).evaluate(df, np.array([100.0]))
    def test_free_nodes_never_negative(self):
        """free_nodes should never go below 0 during simulation."""
        df = _make_df([
            {"job_id": i, "submit_time": i * 2, "num_procs": 30,
             "walltime": 100, "runtime": 80}
            for i in range(10)
        ])
        ev = EASYBackfillEvaluator(num_nodes=100, collect_timeseries=True)
        m = ev.evaluate(df, t_safe=df["walltime"].values)

        ts = m.extra["timeseries"]
        assert (ts["free_nodes"] >= 0).all(), \
            f"free_nodes went negative: min={ts['free_nodes'].min()}"

    def test_utilization_bounded(self):
        """Utilization should be between 0 and 100%."""
        df = _make_df([
            {"job_id": i, "submit_time": i * 5, "num_procs": 20,
             "walltime": 200, "runtime": 100}
            for i in range(5)
        ])
        ev = EASYBackfillEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=df["walltime"].values)

        assert 0 <= m.utilization <= 100, f"Utilization={m.utilization}"





class TestShadow:
    def test_backfilled_job_respects_shadow(self):
        """Backfilled job's walltime must not extend past shadow time
        (unless using excess nodes — condition 2)."""





        df = _make_df([
            {"job_id": 1, "submit_time": 0,  "num_procs": 80, "walltime": 200, "runtime": 200},
            {"job_id": 2, "submit_time": 5,  "num_procs": 90, "walltime": 300, "runtime": 100},
            {"job_id": 3, "submit_time": 10, "num_procs": 20, "walltime": 150, "runtime": 100},
            {"job_id": 4, "submit_time": 15, "num_procs": 20, "walltime": 250, "runtime": 100},
        ])
        ev = EASYBackfillEvaluator(num_nodes=100, collect_timeseries=True)
        m = ev.evaluate(df, t_safe=df["walltime"].values)

        ts = m.extra["timeseries"]
        assert m.nb_jobs_success == 4
        assert m.nb_jobs_rejected == 0





class TestMetrics:
    def test_makespan_computation(self):
        """Makespan = last_finish - first_submit for non-rejected jobs."""
        df = _make_df([
            {"job_id": 1, "submit_time": 100, "num_procs": 10, "walltime": 50, "runtime": 50},
            {"job_id": 2, "submit_time": 200, "num_procs": 10, "walltime": 50, "runtime": 50},
        ])
        ev = EASYBackfillEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=df["walltime"].values)



        assert abs(m.makespan - 150.0) < 1e-6

    def test_success_rate(self):
        """Success rate = successful / total * 100."""
        df = _make_df([
            {"job_id": 1, "submit_time": 0, "num_procs": 10, "walltime": 100, "runtime": 50},
            {"job_id": 2, "submit_time": 0, "num_procs": 10, "walltime": 100, "runtime": 200},
        ])


        ev = EASYBackfillEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=np.array([100.0, 100.0]))

        assert abs(m.success_rate - 50.0) < 1e-6

    def test_empty_trace(self):
        """Empty trace should not crash."""
        df = _make_df([])
        ev = EASYBackfillEvaluator(num_nodes=100)
        m = ev.evaluate(df, t_safe=np.array([]))

        assert m.utilization == 0.0
        assert m.makespan == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
