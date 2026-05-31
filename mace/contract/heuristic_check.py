"""Validate a contract the way MACE actually uses it: generate a real heuristic
and run it through the framework. If a solver can read an instance (I), return a
solution of the right shape (O), and have it scored by the tools (T), the
contract works end to end. This is the 'write a heuristic and run it' check."""
from __future__ import annotations

from mace.evolution.operators._common import build_prompt, extract_python
from mace.evolution.smoke_test import smoke_test


def _gen_heuristic(llm_client, spec, hint: str = "") -> str:
    body = (
        "# Task — write a SIMPLE solver to exercise this contract\n"
        "Write a `solve(instance, tools, time_limit_s)` that returns a FEASIBLE "
        "solution dict. FEASIBILITY IS THE ONLY GOAL — a suboptimal but feasible "
        "solution is SUCCESS; an infeasible one is FAILURE. Respect EVERY "
        "constraint. Build the solution constructively and, before returning, call "
        "`ok, msg = tools['is_feasible'](sol)`; if not ok, read `msg` and REPAIR the "
        "solution (reassign/adjust) and re-check, looping until it is feasible. "
        "Optimality does not matter. Use `tools['objective']` and any other listed "
        "tools as needed.\n"
        + (hint or "")
    )
    prompt = build_prompt(spec, body)
    return extract_python(llm_client.chat(prompt))


def heuristic_passes(spec, llm_client, instance_path: str, hint: str = "",
                     tries: int = 2, time_limit_s: float = 30.0):
    """Generate up to `tries` heuristics; return (ok, err, last_code). ok=True as
    soon as one passes the framework smoke_test on `instance_path`."""
    err = "no heuristic generated"
    code = ""
    for _ in range(tries):
        try:
            code = _gen_heuristic(llm_client, spec, hint)
        except Exception as e:
            err = f"heuristic generation raised: {type(e).__name__}: {e}"
            continue
        passed, err = smoke_test(code, spec, instance_path, time_limit_s=time_limit_s)
        if passed:
            return True, None, code
    return False, err, code
