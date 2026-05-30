"""Smoke-test a candidate solve_code string on one small instance.

Used to catch syntax errors, missing imports, and basic feasibility issues
*before* paying the full evaluation cost on M training instances.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mace.framework import ProblemSpec, run_solve  # noqa: E402


def smoke_test(
    solve_code: str,
    spec: ProblemSpec,
    smoke_instance_path: str,
    time_limit_s: float = 30.0,
    spec_module_path: Optional[str] = None,
    use_subprocess: bool = False,
    hard_kill_slack: float = 2.0,
) -> Tuple[bool, Optional[str]]:
    """exec solve_code in a fresh namespace; run on one instance via framework.

    Returns (passed, error_msg).

    When use_subprocess=True and spec_module_path is set, the candidate is
    executed inside a fresh subprocess that is hard-killed after
    `time_limit_s * hard_kill_slack` -- protects the parent against runaway
    loops in LLM-generated code.
    """
    if use_subprocess and spec_module_path:
        # Lazy import to avoid circular dep (evolve imports from smoke_test? no, but be safe).
        from mace.evolution.evolve import evaluate_one_subprocess
        cost, info = evaluate_one_subprocess(
            spec_module_path, solve_code, smoke_instance_path,
            T_max=time_limit_s, hard_kill_slack=hard_kill_slack,
        )
        if cost >= 1e10:
            status = info.get("status", "fail") if isinstance(info, dict) else "fail"
            msg = info.get("msg", str(info)) if isinstance(info, dict) else str(info)
            return False, f"{status}: {msg}"
        # cost finite -> feasible
        return True, None

    # In-process fallback (kept for back-compat / tests). DO NOT use with
    # untrusted LLM-generated code: a runaway loop will hang the parent.
    ns: dict = {}
    try:
        compiled = compile(solve_code, "<candidate>", "exec")
        exec(compiled, ns)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"
    except Exception as e:
        return False, f"exec raised: {type(e).__name__}: {e}"

    solve_fn = ns.get("solve")
    if solve_fn is None or not callable(solve_fn):
        return False, "no callable `solve` defined"

    try:
        instance = spec.load_data(smoke_instance_path)
    except Exception as e:
        return False, f"load_data failed: {type(e).__name__}: {e}"

    result = run_solve(spec, instance, solve_fn, time_limit_s=time_limit_s)
    if not result.feasible:
        return False, f"infeasible: {result.error_msg}"
    if result.objective is None or not (result.objective == result.objective):  # NaN check
        return False, f"non-finite objective: {result.objective}"
    return True, None
