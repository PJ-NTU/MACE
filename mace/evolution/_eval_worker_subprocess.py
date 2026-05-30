"""Standalone subprocess worker for a single evaluate_one call.

Reads JSON from stdin:
  {"spec_module_path": str, "code": str, "instance_path": str, "T_max": float}
Writes JSON to stdout (last line):
  {"cost": float, "info": dict}

The parent process times out the subprocess via subprocess.run(timeout=...).
"""
from __future__ import annotations
import sys
import json
import importlib
from pathlib import Path


def _bootstrap_sys_path():
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def main():
    _bootstrap_sys_path()
    PENALTY = 1e10
    try:
        data = json.load(sys.stdin)
    except Exception as e:
        print(json.dumps({"cost": PENALTY,
                          "info": {"status": "input_parse_error", "msg": str(e)}}))
        return

    spec_module_path = data.get("spec_module_path")
    code = data.get("code")
    instance_path = data.get("instance_path")
    T_max = float(data.get("T_max", 60.0))

    try:
        module = importlib.import_module(spec_module_path)
        spec = module.SPEC
    except Exception as e:
        print(json.dumps({"cost": PENALTY,
                          "info": {"status": "spec_import_error",
                                   "msg": f"{type(e).__name__}: {e}"}}))
        return

    try:
        from mace.evolution.evolve import evaluate_one
    except Exception as e:
        print(json.dumps({"cost": PENALTY,
                          "info": {"status": "evolve_import_error",
                                   "msg": f"{type(e).__name__}: {e}"}}))
        return

    try:
        cost, info = evaluate_one(spec, code, instance_path, T_max)
    except Exception as e:
        print(json.dumps({"cost": PENALTY,
                          "info": {"status": "worker_exception",
                                   "msg": f"{type(e).__name__}: {e}"}}))
        return

    try:
        print(json.dumps({"cost": float(cost), "info": info}))
    except Exception as e:
        # info may contain non-serializable values; coerce
        safe = {k: str(v) for k, v in info.items()}
        print(json.dumps({"cost": float(cost), "info": safe}))


if __name__ == "__main__":
    main()
