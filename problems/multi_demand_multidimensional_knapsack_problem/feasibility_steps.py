"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    x = solution.get("x")
    if x is None:
        return False, "solution missing 'x' key"

    if not isinstance(x, list):
        return False, f"'x' must be list, got {type(x).__name__}"
    if len(x) != n:
        return False, f"'x' length {len(x)} != n={n}"
    if any(val not in (0, 1) for val in x):
        return False, "'x' must be binary (entries must be 0 or 1)"

    for i in range(m):
        lhs = sum(A_leq[i][j] * x[j] for j in range(n))
        if lhs > b_leq[i]:
            return False, f"<= constraint {i+1} violated: lhs={lhs} > rhs={b_leq[i]}"

    for i in range(q):
        lhs = sum(A_geq[i][j] * x[j] for j in range(n))
        if lhs < b_geq[i]:
            return False, f">= constraint {i+1} violated: lhs={lhs} < rhs={b_geq[i]}"

    return True, None
'''
