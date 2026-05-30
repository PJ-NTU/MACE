"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    schedule = solution.get("schedule")
    if schedule is None:
        return False, "solution missing 'schedule' key"

    if not isinstance(schedule, (list, tuple)):
        return False, f"'schedule' must be list, got {type(schedule).__name__}"

    n = len(jobs)

    if len(schedule) != n:
        return False, f"schedule length {len(schedule)} != n={n}"

    if any(not isinstance(x, int) for x in schedule):
        return False, "schedule entries must be int"

    if sorted(schedule) != list(range(1, n + 1)):
        return False, f"schedule must be a permutation of 1 to {n}"

    return True, None
'''
