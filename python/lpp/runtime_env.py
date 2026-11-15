"""Size thread pools to the CPUs this process may actually use.

In containers (e.g. the SCNet DCU notebook: 15-core quota on a 128-core host)
os.cpu_count() reports the host, and PyTorch / OpenMP then start one thread per host
core, which oversubscribes the quota and slows everything down. The real budget is
the cgroup CPU quota, or failing that the scheduler affinity mask.
"""
from __future__ import annotations

import math
import os
from pathlib import Path


def available_cpus() -> int:
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for f in ("/sys/fs/cgroup/cpu.max",):                       # cgroup v2: "<quota> <period>" or "max"
        try:
            quota, period = Path(f).read_text().split()
            if quota != "max":
                n = min(n, max(1, math.floor(int(quota) / int(period))))
        except (OSError, ValueError):
            pass
    try:                                                         # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            n = min(n, max(1, quota // period))
    except (OSError, ValueError):
        pass
    return n


def configure_threads(reserve_for_workers: int = 0) -> int:
    """Set OpenMP / MKL / torch intra-op threads to the CPU budget minus data workers.

    Respects an explicit OMP_NUM_THREADS. Returns the thread count used.
    """
    import torch
    if "OMP_NUM_THREADS" in os.environ:
        n = int(os.environ["OMP_NUM_THREADS"])
    else:
        n = max(1, available_cpus() - reserve_for_workers)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(var, str(n))
    torch.set_num_threads(n)
    return n


def default_workers(cap: int = 8) -> int:
    """DataLoader workers: about half the CPU budget, at most `cap`."""
    return max(0, min(cap, available_cpus() // 2))
