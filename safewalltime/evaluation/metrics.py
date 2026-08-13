from __future__ import annotations
"""Prediction and scheduling evaluation metrics.

* **Job success rate** -- percentage of jobs where t_safe >= actual runtime.
* **Kill rate** -- percentage of jobs terminated (t_safe < actual runtime).
* **Total waste** -- sum of (t_safe - runtime) for successful jobs (seconds).
* **Resource utilization** -- time_computing / (time_computing + time_idle).

Evaluators:

* ``SchedulingEvaluator`` -- abstract base class for scheduling simulation.
* ``EASYBackfillEvaluator`` -- discrete-event EASY backfill simulation
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from itertools import islice
from typing import Optional
import heapq

import numpy as np
import pandas as pd




@dataclass
class PredictionMetrics:
    """Aggregate prediction-level metrics.

    ``coverage_rate`` measures the fraction of jobs whose predicted safe
    runtime (t_safe) exceeds the actual runtime — i.e. jobs that would NOT
    be killed.  This is distinct from ``SchedulingMetrics.success_rate``
    which counts jobs that completed in the discrete-event simulation
    (excluding both killed and rejected jobs).

    Attributes:
        coverage_rate: Fraction of jobs where t_safe >= runtime (%).
        underestimation_rate: Fraction of jobs where t_safe < runtime (%).
                              Equal to ``100 - coverage_rate``.
        total_waste: Total overprediction across all covered jobs (seconds).
        mean_waste: Mean overprediction per covered job (seconds).
        mae: Mean absolute error (seconds).
    """

    coverage_rate: float
    underestimation_rate: float
    total_waste: float
    mean_waste: float
    mae: float


@dataclass
class SchedulingMetrics:
    """Scheduling-level metrics

    Attributes:
        utilization: time_computing / (time_computing + time_idle) * 100 (%).
        makespan: max(finish_time) - min(submit_time) (seconds).
        mean_slowdown: Average bounded slowdown (tau=10s).
        mean_wait_time: Average waiting time (seconds).
        mean_turnaround_time: Average turnaround time (seconds).
        nb_jobs_success: Number of successfully completed jobs.
        nb_jobs_killed: Number of jobs killed (walltime reached).
        nb_jobs_rejected: Number of jobs rejected (insufficient resources).
        success_rate: nb_jobs_success / nb_jobs * 100 (%).
        time_computing: Cumulative machine-seconds in computing state.
        time_idle: Cumulative machine-seconds in idle state.
        extra: Additional simulator-specific metrics.
    """

    utilization: float
    makespan: float
    mean_slowdown: float = 0.0
    mean_wait_time: float = 0.0
    mean_turnaround_time: float = 0.0
    nb_jobs_success: int = 0
    nb_jobs_killed: int = 0
    nb_jobs_rejected: int = 0
    success_rate: float = 0.0
    time_computing: float = 0.0
    time_idle: float = 0.0
    extra: dict = field(default_factory=dict)




def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> PredictionMetrics:
    """Compute prediction-level metrics from actual and predicted runtimes.

    Args:
        y_true: Actual runtimes (seconds).
        y_pred: Predicted safe runtimes (t_safe, seconds).

    Returns:
        PredictionMetrics.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = len(y_true)
    if n == 0:
        return PredictionMetrics(
            coverage_rate=0.0, underestimation_rate=0.0,
            total_waste=0.0, mean_waste=0.0, mae=0.0,
        )

    covered = y_pred >= y_true
    n_covered = int(covered.sum())

    coverage_rate = n_covered / n * 100.0
    underestimation_rate = 100.0 - coverage_rate


    waste = np.where(covered, y_pred - y_true, 0.0)
    total_waste = float(waste.sum())
    mean_waste = float(waste[covered].mean()) if n_covered > 0 else 0.0

    mae = float(np.abs(y_pred - y_true).mean())


    return PredictionMetrics(
        coverage_rate=coverage_rate,
        underestimation_rate=underestimation_rate,
        total_waste=total_waste,
        mean_waste=mean_waste,
        mae=mae,
    )




class SchedulingEvaluator(ABC):
    """Abstract base class for scheduling evaluation.

    Subclass this to integrate with simulators or to implement analytical scheduling models.
    """

    @abstractmethod
    def evaluate(
        self,
        trace_df: pd.DataFrame,
        t_safe: np.ndarray,
    ) -> SchedulingMetrics:
        """Run scheduling evaluation.

        Args:
            trace_df: Test-set DataFrame with canonical columns.
            t_safe: Per-job safe runtime estimates (seconds, original scale).

        Returns:
            SchedulingMetrics.
        """
        ...

class FCFSEvaluator(SchedulingEvaluator):
    """Discrete-event FCFS (First Come First Served) simulation — no backfill.

    Baseline scheduler: jobs are started strictly in submission order.
    A job can only start when all jobs submitted before it have started
    (or been rejected).  No backfilling of smaller later jobs.

    Args:
        num_nodes: Total number of computing nodes.
        bounded_slowdown_tau: Threshold for bounded slowdown. Default: 10s.
        collect_timeseries: If True, record per-event snapshots.
    """

    def __init__(
        self,
        num_nodes: Optional[int] = None,
        bounded_slowdown_tau: float = 10.0,
        collect_timeseries: bool = False,
    ) -> None:
        self.num_nodes = num_nodes
        self.tau = bounded_slowdown_tau
        self.collect_timeseries = collect_timeseries

    def evaluate(
        self,
        trace_df: pd.DataFrame,
        t_safe: np.ndarray,
    ) -> SchedulingMetrics:
        t_safe = np.asarray(t_safe, dtype=np.float64)
        runtime = trace_df["runtime"].values.astype(np.float64)
        submit = trace_df["submit_time"].values.astype(np.float64)
        num_procs = (
            trace_df["num_procs"].values.astype(np.int64)
            if "num_procs" in trace_df.columns
            else np.ones(len(runtime), dtype=np.int64)
        )

        n = len(runtime)
        if t_safe.ndim != 1 or len(t_safe) != n:
            raise ValueError("t_safe must be a one-dimensional value for every job.")
        if not np.isfinite(t_safe).all() or np.any(t_safe <= 0):
            raise ValueError("t_safe values must be finite and strictly positive.")
        if not np.isfinite(runtime).all() or np.any(runtime < 0):
            raise ValueError("runtime values must be finite and non-negative.")
        if not np.isfinite(submit).all():
            raise ValueError("submit_time values must be finite.")
        if np.any(num_procs <= 0):
            raise ValueError("num_procs values must be strictly positive.")
        if n == 0:
            return SchedulingMetrics(utilization=0.0, makespan=0.0)

        num_nodes = (
            self.num_nodes if self.num_nodes is not None else int(num_procs.max())
        )
        if num_nodes <= 0:
            raise ValueError("num_nodes must be strictly positive.")

        ts_time: list[float] = []
        ts_util: list[float] = []
        ts_queue: list[int] = []
        ts_running: list[int] = []
        ts_free: list[int] = []

        order = np.argsort(submit, kind="stable")
        submit = submit[order]
        runtime = runtime[order]
        t_safe_sorted = t_safe[order]
        num_procs_sorted = num_procs[order]

        start_time = np.full(n, -1.0)
        finish_time = np.full(n, -1.0)
        killed = np.zeros(n, dtype=bool)
        rejected = np.zeros(n, dtype=bool)

        running_heap: list[tuple[float, int, int]] = []
        free_nodes = num_nodes
        wait_queue: deque[int] = deque()

        def _free_completed(current_time: float) -> None:
            nonlocal free_nodes
            while running_heap and running_heap[0][0] <= current_time:
                _, nodes, _ = heapq.heappop(running_heap)
                free_nodes += nodes

        def _start_job(idx: int, current_time: float) -> None:
            nonlocal free_nodes
            nodes = int(num_procs_sorted[idx])
            walltime = t_safe_sorted[idx]
            actual = runtime[idx]
            free_nodes -= nodes
            start_time[idx] = current_time
            if actual > walltime:
                exec_time = walltime
                killed[idx] = True
            else:
                exec_time = actual
            finish_time[idx] = current_time + exec_time
            heapq.heappush(running_heap, (current_time + exec_time, nodes, idx))

        job_ptr = 0
        sim_time = 0.0

        while job_ptr < n or wait_queue or running_heap:
            next_times = []
            if job_ptr < n:
                next_times.append(submit[job_ptr])
            if running_heap:
                next_times.append(running_heap[0][0])
            if not next_times:
                break
            sim_time = min(next_times)

            _free_completed(sim_time)

            while job_ptr < n and submit[job_ptr] <= sim_time:
                nodes_needed = int(num_procs_sorted[job_ptr])
                if nodes_needed > num_nodes:
                    rejected[job_ptr] = True
                    finish_time[job_ptr] = submit[job_ptr]
                    start_time[job_ptr] = submit[job_ptr]
                else:
                    wait_queue.append(job_ptr)
                job_ptr += 1


            if wait_queue:
                while wait_queue:
                    first_job = wait_queue[0]
                    first_nodes = int(num_procs_sorted[first_job])
                    if first_nodes <= free_nodes:
                        _start_job(first_job, sim_time)
                        wait_queue.popleft()
                    else:
                        break

            if self.collect_timeseries:
                used = num_nodes - free_nodes
                ts_time.append(sim_time)
                ts_util.append(used / num_nodes * 100.0 if num_nodes > 0 else 0.0)
                ts_queue.append(len(wait_queue))
                ts_running.append(len(running_heap))
                ts_free.append(free_nodes)


        completed_mask = ~rejected
        started_mask = start_time >= 0
        valid = completed_mask & started_mask

        n_valid = int(valid.sum())
        n_success = int((valid & ~killed).sum())
        n_killed_total = int(killed.sum())
        n_rejected_total = int(rejected.sum())

        if n_valid == 0:
            return SchedulingMetrics(
                utilization=0.0, makespan=0.0,
                nb_jobs_rejected=n_rejected_total,
            )

        makespan = float(finish_time[valid].max() - submit[valid].min())
        if makespan <= 0:
            makespan = 1.0

        exec_times = finish_time[valid] - start_time[valid]
        nodes_valid = num_procs_sorted[valid].astype(np.float64)
        time_computing = float((exec_times * nodes_valid).sum())

        total_capacity = num_nodes * makespan
        utilization = (
            time_computing / total_capacity * 100.0
            if total_capacity > 0 else 0.0
        )
        time_idle = total_capacity - time_computing

        wait = start_time[valid] - submit[valid]
        turnaround = finish_time[valid] - submit[valid]
        mean_wait = float(wait.mean()) if n_valid > 0 else 0.0
        mean_turnaround = float(turnaround.mean()) if n_valid > 0 else 0.0

        if n_valid > 0:
            bsld = turnaround / np.maximum(exec_times, self.tau)
            bsld = np.maximum(bsld, 1.0)
            mean_slowdown = float(bsld.mean())
        else:
            mean_slowdown = 0.0

        extra: dict = {}
        if self.collect_timeseries:
            extra["timeseries"] = {
                "time": np.array(ts_time),
                "utilization": np.array(ts_util),
                "queue_length": np.array(ts_queue, dtype=np.int64),
                "running_jobs": np.array(ts_running, dtype=np.int64),
                "free_nodes": np.array(ts_free, dtype=np.int64),
            }

        return SchedulingMetrics(
            utilization=utilization,
            makespan=makespan,
            mean_slowdown=mean_slowdown,
            mean_wait_time=mean_wait,
            mean_turnaround_time=mean_turnaround,
            nb_jobs_success=n_success,
            nb_jobs_killed=n_killed_total,
            nb_jobs_rejected=n_rejected_total,
            success_rate=(n_success / n * 100.0 if n > 0 else 0.0),
            time_computing=time_computing,
            time_idle=max(time_idle, 0.0),
            extra=extra,
        )


class EASYBackfillEvaluator(SchedulingEvaluator):
    """Discrete-event EASY backfill simulation.

    Simulates the EASY (Extensible Argonne Scheduling sYstem) backfill algorithm

    Algorithm:
        1. Jobs arrive sorted by submit_time (FCFS queue).
        2. Try to start queued jobs in FCFS order (priority phase).
        3. When first job cannot start, create a reservation (shadow time)
           computed from running jobs' **walltime** end times (not actual).
        4. Backfill lower-priority jobs whose walltime fits before shadow.
        5. Jobs exceeding their walltime (t_safe) are killed.

    Args:
        num_nodes: Total number of computing nodes in the simulated cluster.
                   If None, inferred as max(num_procs) from the trace.
        bounded_slowdown_tau: Threshold for bounded slowdown. Default: 10s.
        collect_per_job: If True, include per-job result arrays in
            ``extra["per_job"]``.  Arrays are mapped back to **original**
            (unsorted) order matching the input trace_df.
    """

    def __init__(
        self,
        num_nodes: Optional[int] = None,
        bounded_slowdown_tau: float = 10.0,
        collect_timeseries: bool = False,
        collect_per_job: bool = False,
    ) -> None:
        self.num_nodes = num_nodes
        self.tau = bounded_slowdown_tau
        self.collect_timeseries = collect_timeseries
        self.collect_per_job = collect_per_job

    def evaluate(
        self,
        trace_df: pd.DataFrame,
        t_safe: np.ndarray,
    ) -> SchedulingMetrics:
        """Run EASY backfill simulation.

        Args:
            trace_df: Test-set DataFrame with canonical columns.
            t_safe: Per-job walltime estimates (seconds, original scale).

        Returns:
            SchedulingMetrics.  If ``collect_timeseries`` is True, the
            ``extra`` dict contains a ``"timeseries"`` key with a dict of
            numpy arrays: ``time``, ``utilization``, ``queue_length``,
            ``running_jobs``, ``free_nodes``.
        """
        t_safe = np.asarray(t_safe, dtype=np.float64)
        runtime = trace_df["runtime"].values.astype(np.float64)
        submit = trace_df["submit_time"].values.astype(np.float64)
        num_procs = (
            trace_df["num_procs"].values.astype(np.int64)
            if "num_procs" in trace_df.columns
            else np.ones(len(runtime), dtype=np.int64)
        )

        n = len(runtime)
        if t_safe.ndim != 1 or len(t_safe) != n:
            raise ValueError("t_safe must be a one-dimensional value for every job.")
        if not np.isfinite(t_safe).all() or np.any(t_safe <= 0):
            raise ValueError("t_safe values must be finite and strictly positive.")
        if not np.isfinite(runtime).all() or np.any(runtime < 0):
            raise ValueError("runtime values must be finite and non-negative.")
        if not np.isfinite(submit).all():
            raise ValueError("submit_time values must be finite.")
        if np.any(num_procs <= 0):
            raise ValueError("num_procs values must be strictly positive.")
        if n == 0:
            return SchedulingMetrics(utilization=0.0, makespan=0.0)

        num_nodes = (
            self.num_nodes if self.num_nodes is not None else int(num_procs.max())
        )
        if num_nodes <= 0:
            raise ValueError("num_nodes must be strictly positive.")


        ts_time: list[float] = []
        ts_util: list[float] = []
        ts_queue: list[int] = []
        ts_running: list[int] = []
        ts_free: list[int] = []


        order = np.argsort(submit, kind="stable")
        submit = submit[order]
        runtime = runtime[order]
        t_safe_sorted = t_safe[order]
        num_procs_sorted = num_procs[order]


        start_time = np.full(n, -1.0)
        finish_time = np.full(n, -1.0)
        killed = np.zeros(n, dtype=bool)
        rejected = np.zeros(n, dtype=bool)



        running_heap: list[tuple[float, int, int]] = []

        running_walltime: dict[int, tuple[float, int]] = {}
        free_nodes = num_nodes
        wait_queue: deque[int] = deque()

        def _free_completed(current_time: float) -> None:
            """Free resources from jobs that have completed by current_time."""
            nonlocal free_nodes
            while running_heap and running_heap[0][0] <= current_time:
                end_t, nodes, idx = heapq.heappop(running_heap)
                free_nodes += nodes
                running_walltime.pop(idx, None)

        def _start_job(idx: int, current_time: float) -> None:
            """Start a job, allocating resources."""
            nonlocal free_nodes
            nodes = int(num_procs_sorted[idx])
            walltime = t_safe_sorted[idx]
            actual = runtime[idx]

            free_nodes -= nodes
            start_time[idx] = current_time


            if actual > walltime:
                exec_time = walltime
                killed[idx] = True
            else:
                exec_time = actual

            actual_end = current_time + exec_time
            walltime_end = current_time + walltime

            finish_time[idx] = actual_end
            heapq.heappush(running_heap, (actual_end, nodes, idx))
            running_walltime[idx] = (walltime_end, nodes)

        def _compute_shadow(first_idx: int) -> tuple[float, int]:
            """Compute shadow time and extra resources at shadow.

            The scheduler does not know actual runtimes, so it estimates
            when resources will be freed based on each running job's
            walltime (= start_time + t_safe). 

            Returns:
                (shadow_time, extra_nodes) where extra_nodes is the number
                of nodes available at shadow_time beyond what the first job
                needs.  These can be used by backfill jobs that finish after
                shadow_time (Mu'alem & Feitelson condition 2).
            """
            needed = int(num_procs_sorted[first_idx])
            if needed > num_nodes:
                return float("inf"), 0


            wt_ends = sorted(running_walltime.values())
            available = free_nodes
            position = 0
            while position < len(wt_ends):
                wt_end = wt_ends[position][0]
                while position < len(wt_ends) and wt_ends[position][0] == wt_end:
                    available += wt_ends[position][1]
                    position += 1
                if available >= needed:
                    return wt_end, available - needed
            return float("inf"), 0


        job_ptr = 0
        sim_time = 0.0

        while job_ptr < n or wait_queue or running_heap:

            next_times = []
            if job_ptr < n:
                next_times.append(submit[job_ptr])
            if running_heap:
                next_times.append(running_heap[0][0])
            if not next_times:
                break
            sim_time = min(next_times)


            _free_completed(sim_time)


            while job_ptr < n and submit[job_ptr] <= sim_time:
                nodes_needed = int(num_procs_sorted[job_ptr])
                if nodes_needed > num_nodes:
                    rejected[job_ptr] = True
                    finish_time[job_ptr] = submit[job_ptr]
                    start_time[job_ptr] = submit[job_ptr]
                else:
                    wait_queue.append(job_ptr)
                job_ptr += 1


            if wait_queue:

                while wait_queue:
                    first_job = wait_queue[0]
                    first_nodes = int(num_procs_sorted[first_job])
                    if first_nodes <= free_nodes:
                        _start_job(first_job, sim_time)
                        wait_queue.popleft()
                    else:
                        break

                if wait_queue:

                    first_job = wait_queue[0]
                    shadow_time, extra_at_shadow = _compute_shadow(first_job)






                    extra_remaining = extra_at_shadow
                    new_wait_queue: deque[int] = deque([first_job])
                    rest = list(islice(wait_queue, 1, None))
                    for idx in rest:
                        nodes_needed = int(num_procs_sorted[idx])
                        walltime = t_safe_sorted[idx]

                        if nodes_needed > free_nodes:
                            new_wait_queue.append(idx)
                            continue

                        estimated_end = sim_time + walltime
                        if estimated_end <= shadow_time:

                            _start_job(idx, sim_time)
                        elif nodes_needed <= extra_remaining:

                            _start_job(idx, sim_time)
                            extra_remaining -= nodes_needed
                        else:
                            new_wait_queue.append(idx)

                    wait_queue = new_wait_queue


            if self.collect_timeseries:
                used = num_nodes - free_nodes
                ts_time.append(sim_time)
                ts_util.append(used / num_nodes * 100.0 if num_nodes > 0 else 0.0)
                ts_queue.append(len(wait_queue))
                ts_running.append(len(running_heap))
                ts_free.append(free_nodes)


        completed_mask = ~rejected
        started_mask = start_time >= 0
        valid = completed_mask & started_mask

        n_valid = int(valid.sum())
        n_success = int((valid & ~killed).sum())
        n_killed_total = int(killed.sum())
        n_rejected_total = int(rejected.sum())

        if n_valid == 0:
            return SchedulingMetrics(
                utilization=0.0, makespan=0.0,
                nb_jobs_rejected=n_rejected_total,
            )


        makespan = float(
            finish_time[valid].max() - submit[valid].min()
        )
        if makespan <= 0:
            makespan = 1.0


        exec_times = finish_time[valid] - start_time[valid]
        nodes_valid = num_procs_sorted[valid].astype(np.float64)
        time_computing = float((exec_times * nodes_valid).sum())


        total_capacity = num_nodes * makespan
        utilization = (
            time_computing / total_capacity * 100.0
            if total_capacity > 0
            else 0.0
        )
        time_idle = total_capacity - time_computing


        wait = start_time[valid] - submit[valid]
        turnaround = finish_time[valid] - submit[valid]

        mean_wait = float(wait.mean()) if n_valid > 0 else 0.0
        mean_turnaround = float(turnaround.mean()) if n_valid > 0 else 0.0


        if n_valid > 0:
            bsld = turnaround / np.maximum(exec_times, self.tau)
            bsld = np.maximum(bsld, 1.0)
            mean_slowdown = float(bsld.mean())
        else:
            mean_slowdown = 0.0

        extra: dict = {}
        if self.collect_timeseries:
            extra["timeseries"] = {
                "time": np.array(ts_time),
                "utilization": np.array(ts_util),
                "queue_length": np.array(ts_queue, dtype=np.int64),
                "running_jobs": np.array(ts_running, dtype=np.int64),
                "free_nodes": np.array(ts_free, dtype=np.int64),
            }

        if self.collect_per_job:

            inv_order = np.argsort(order)
            pj_wait = np.where(start_time >= 0, start_time - submit, np.nan)
            pj_turnaround = np.where(
                finish_time >= 0, finish_time - submit, np.nan,
            )
            pj_exec = np.where(
                finish_time >= 0, finish_time - start_time, np.nan,
            )
            pj_bsld = np.where(
                finish_time >= 0,
                np.maximum(
                    pj_turnaround / np.maximum(pj_exec, self.tau), 1.0,
                ),
                np.nan,
            )
            extra["per_job"] = {
                "killed": killed[inv_order],
                "rejected": rejected[inv_order],
                "wait_time": pj_wait[inv_order],
                "turnaround": pj_turnaround[inv_order],
                "slowdown": pj_bsld[inv_order],
            }

        return SchedulingMetrics(
            utilization=utilization,
            makespan=makespan,
            mean_slowdown=mean_slowdown,
            mean_wait_time=mean_wait,
            mean_turnaround_time=mean_turnaround,
            nb_jobs_success=n_success,
            nb_jobs_killed=n_killed_total,
            nb_jobs_rejected=n_rejected_total,
            success_rate=(
                n_success / n * 100.0 if n > 0 else 0.0
            ),
            time_computing=time_computing,
            time_idle=max(time_idle, 0.0),
            extra=extra,
        )
