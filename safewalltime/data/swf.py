from __future__ import annotations
"""Standard Workload Format (SWF) trace loader.

SWF is the de-facto format for HPC workload traces (Feitelson et al., 2014).
Each non-comment line has 18 whitespace-separated fields.

Reference: https://www.cs.huji.ac.il/labs/parallel/workload/swf.html

This loader extracts the six features used in the UARP paper:
    job_id, submit_time, queue, num_procs, walltime, memory
plus the target ``runtime``.
"""

import numpy as np
import pandas as pd

from .base import JobTrace, TraceLoader


SWF_COLUMNS = [
    "job_id",
    "submit_time",
    "wait_time",
    "runtime",
    "num_alloc_procs",
    "avg_cpu_time",
    "used_memory",
    "num_procs",
    "walltime",
    "memory",
    "status",
    "user_id",
    "group_id",
    "executable",
    "queue",
    "partition",
    "preceding_job",
    "think_time",
]


UARP_FEATURES = [
    "job_id",
    "submit_time",
    "user_id",
    "queue",
    "num_procs",
    "walltime",
    "memory",
]


class SWFLoader(TraceLoader):
    """Load traces in Standard Workload Format (.swf).

    Args:
        min_runtime: Discard jobs with runtime below this threshold (seconds).
        filter_status: If True, remove jobs with status == 0 (failed).
                       Preserves status == -1 (info not available) per SWF convention.
    """

    def __init__(
        self,
        min_runtime: float = 0.0,
        filter_status: bool = True,
    ) -> None:
        super().__init__(min_runtime=min_runtime)
        self.filter_status = filter_status

    def load(self, path: str) -> JobTrace:
        """Parse an SWF file and return a validated JobTrace.

        Args:
            path: Path to the ``.swf`` file.

        Returns:
            JobTrace with canonical column names.
        """
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                fields = line.split()
                if len(fields) < 18:
                    continue
                rows.append([_parse_field(f) for f in fields[:18]])

        df = pd.DataFrame(rows, columns=SWF_COLUMNS)



        if self.filter_status:
            df = df[df["status"] != 0]


        df = df[df["runtime"] > 0]







        for col in ("num_procs", "memory", "queue", "user_id"):
            df[col] = df[col].clip(lower=0)
        df["walltime"] = df["walltime"].where(df["walltime"] >= 0, np.nan)

        return self._validate(
            df,
            format_name="SWF",
            metadata={"source_path": path},
        )


def _parse_field(s: str) -> float:
    """Parse a single SWF field, treating non-numeric as -1."""
    try:
        return float(s)
    except ValueError:
        return -1.0
