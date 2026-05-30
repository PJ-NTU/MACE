"""Batch eval worker — runs many (code × instance_path) cells in parallel.

Reads JSON from stdin:
  {
    "spec_module_path": str,
    "T_max":  float,
    "n_workers": int (default 8),
    "hard_kill_slack": float (default 2.0),
    "code": str            # OPTIONAL: if present, all cells share this code
    "cells": [             # list of cells to evaluate
        {"code": str (optional if top-level code set),
         "instance_path": str,
         "tag": str (optional, echoed back) },
        ...
    ]
  }

Writes JSON to stdout (last line is a single-line JSON array):
  [{"cost": float, "info": dict, "tag": str (or None)}, ...]   (preserves cell order)

Designed to be invoked over SSH from another machine: the parent ssh's the
remote `python -m mace.evolution._eval_worker_batch`, pipes JSON in, reads JSON out.
"""
from __future__ import annotations
import sys
import json
import importlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError


def _bootstrap_sys_path():
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


PENALTY = 1e10


def _eval_one_worker(args):
    """Top-level (picklable) function the worker pool calls per cell."""
    spec_module_path, code, instance_path, T_max = args
    try:
        module = importlib.import_module(spec_module_path)
        spec = module.SPEC
    except Exception as e:
        return (PENALTY, {"status": "spec_import_error",
                            "msg": f"{type(e).__name__}: {e}"})
    try:
        from mace.evolution.evolve import evaluate_one
        cost, info = evaluate_one(spec, code, instance_path, T_max)
        # Coerce info to JSON-safe
        safe_info = {}
        for k, v in info.items():
            try:
                json.dumps(v); safe_info[k] = v
            except Exception:
                safe_info[k] = str(v)
        return (float(cost), safe_info)
    except Exception as e:
        return (PENALTY, {"status": "worker_exception",
                            "msg": f"{type(e).__name__}: {e}"})


def main():
    _bootstrap_sys_path()
    try:
        data = json.load(sys.stdin)
    except Exception as e:
        print(json.dumps([{"cost": PENALTY,
                              "info": {"status": "input_parse_error", "msg": str(e)},
                              "tag": None}]))
        return

    spec_module_path = data["spec_module_path"]
    T_max = float(data.get("T_max", 60.0))
    n_workers = int(data.get("n_workers", 8))
    hard_kill_slack = float(data.get("hard_kill_slack", 2.0))
    shared_code = data.get("code")
    cells = data["cells"]

    args_list = []
    tags = []
    for c in cells:
        code = c.get("code") if c.get("code") is not None else shared_code
        if code is None:
            raise ValueError("cell missing 'code' and no top-level 'code'")
        args_list.append((spec_module_path, code, c["instance_path"], T_max))
        tags.append(c.get("tag"))

    # Per-cell timeout budget = T_max * slack.  In practice the inner evaluate_one
    # already respects T_max via the framework's run_solve (which the heuristic
    # is expected to honor); the slack is for outliers.  Worker pool itself does
    # NOT kill rogue cells — we trust evaluate_one's penalty handling.
    n = len(args_list)
    n_workers = max(1, min(n_workers, n))

    results = []
    if n_workers == 1:
        for a in args_list:
            results.append(_eval_one_worker(a))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for r in pool.map(_eval_one_worker, args_list):
                results.append(r)

    out = [{"cost": cost, "info": info, "tag": tag}
            for (cost, info), tag in zip(results, tags)]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
