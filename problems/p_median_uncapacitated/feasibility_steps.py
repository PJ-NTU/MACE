"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    medians = solution.get("medians")
    if medians is None:
        return False, "solution missing 'medians' key"

    if not isinstance(medians, list):
        return False, f"'medians' must be list, got {type(medians).__name__}"

    if len(medians) != p:
        return False, f"'medians' must have exactly {p} elements, got {len(medians)}"

    if any(not isinstance(m, int) for m in medians):
        return False, "each median must be an integer"

    if any(m < 1 or m > n for m in medians):
        return False, f"each median must be in range [1, n] where n={n}"

    if len(set(medians)) != p:
        return False, "medians must be distinct"

    INF = float('inf')
    for i in range(n):
        if all(dist[i][m - 1] == INF for m in medians):
            return False, f"vertex {i + 1} is unreachable from all chosen medians"

    return True, None
'''
