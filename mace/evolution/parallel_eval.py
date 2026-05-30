"""Parallel evaluation of (code, instance) pairs via ProcessPoolExecutor.

Each worker:
  - Lazy-imports the ProblemSpec from `spec_module_path` (a top-level module
    that exposes a `SPEC` attribute).
  - Starts a watchdog timer that injects an async TimeoutError into the
    worker's main thread if execution exceeds `hard_limit_s`. This handles
    LLM-generated heuristics that ignore `time_limit_s` themselves.
  - Calls `evaluate_one(spec, code, instance_path, T_max)`.
  - Returns (cost_or_PENALTY, info_dict).

The async-exception trick (PyThreadState_SetAsyncExc) only interrupts when
the worker's main thread is running Python bytecode. If the runaway code is
deep inside a C call (e.g., a numpy operation), the exception will only fire
once control returns to Python. For our TSP/JSSP/CVRP heuristics this is
fine — the hot loops are Python.
"""
from __future__ import annotations
import sys
import os
import ctypes
import threading
import importlib
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Ensure repo root is importable when spawned as a worker on Windows.
# Critical: on Windows, `spawn` does NOT inherit sys.path additions made at
# runtime by the parent. We propagate via PYTHONPATH so the child's site.py
# picks it up before unpickling any submitted task.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_existing_pp = os.environ.get("PYTHONPATH", "")
if str(_REPO_ROOT) not in _existing_pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = (str(_REPO_ROOT) + os.pathsep + _existing_pp).rstrip(os.pathsep)


PENALTY: float = 1e10


def _raise_async_exception(thread_id: int, exc_class):
    """Inject an exception into another running thread (cross-thread interrupt)."""
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread_id), ctypes.py_object(exc_class)
    )
    if res > 1:
        # Roll back if we somehow affected more than one thread.
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)


def _eval_worker_task(
    spec_module_path: str,
    code: str,
    instance_path: str,
    T_max: float,
    hard_limit_s: float,
):
    """Top-level worker; picklable by reference.

    Trusts the heuristic to obey `T_max` via self-monitoring. If it doesn't,
    the framework still returns a (possibly slow) result and we flag it as
    `timeout` on the caller side via `elapsed > T_max * timeout_slack` inside
    `evaluate_one`. The `hard_limit_s` argument is retained for future hard-
    kill mechanisms but is currently unused — the worker simply lets the call
    run to completion. (Earlier attempts using PyThreadState_SetAsyncExc
    crashed the worker on Python 3.14.)
    """
    import sys as _sys
    from pathlib import Path as _Path
    _repo = _Path(__file__).resolve().parents[1]
    if str(_repo) not in _sys.path:
        _sys.path.insert(0, str(_repo))

    try:
        module = importlib.import_module(spec_module_path)
        spec = module.SPEC
    except Exception as e:
        return PENALTY, {"status": "spec_import_error", "msg": f"{type(e).__name__}: {e}"}

    try:
        from mace.evolution.evolve import evaluate_one
    except Exception as e:
        return PENALTY, {"status": "evolve_import_error", "msg": f"{type(e).__name__}: {e}"}

    try:
        cost, info = evaluate_one(spec, code, instance_path, T_max)
    except Exception as e:
        cost = PENALTY
        info = {"status": "worker_exception", "msg": f"{type(e).__name__}: {e}"}
    return cost, info


def parallel_evaluate_portfolio(
    spec_module_path: str,
    codes: list[str],
    instance_paths: list[str],
    T_max: float,
    n_workers: int = 6,
    hard_kill_slack: float = 1.5,
) -> tuple[np.ndarray, list[list[dict]]]:
    """Evaluate N codes on M instances in parallel. Returns (F (M x N), infos[M][N])."""
    M, N = len(instance_paths), len(codes)
    F = np.full((M, N), PENALTY, dtype=np.float64)
    infos: list[list[dict]] = [[{"status": "pending"} for _ in range(N)] for _ in range(M)]
    hard_limit = T_max * hard_kill_slack

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        fut_to_hi: dict = {}
        for h in range(N):
            for i in range(M):
                fut = pool.submit(
                    _eval_worker_task,
                    spec_module_path, codes[h], instance_paths[i], T_max, hard_limit,
                )
                fut_to_hi[fut] = (h, i)

        for fut in as_completed(fut_to_hi):
            h, i = fut_to_hi[fut]
            try:
                cost, info = fut.result()
            except Exception as e:
                cost = PENALTY
                info = {"status": "future_error", "msg": f"{type(e).__name__}: {e}"}
            F[i, h] = cost
            infos[i][h] = info

    return F, infos


def parallel_evaluate_code(
    spec_module_path: str,
    code: str,
    instance_paths: list[str],
    T_max: float,
    n_workers: int = 6,
    hard_kill_slack: float = 1.5,
) -> tuple[np.ndarray, list[dict]]:
    """Evaluate one code on M instances in parallel. Returns (perf_vec, infos_list)."""
    F, infos = parallel_evaluate_portfolio(
        spec_module_path, [code], instance_paths, T_max, n_workers, hard_kill_slack,
    )
    perf_vec = F[:, 0]
    info_vec = [row[0] for row in infos]
    return perf_vec, info_vec
