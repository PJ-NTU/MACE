"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    x = solution.get("x")
    if x is None:
        return False, "solution missing 'x' key"

    if not isinstance(x, (list, tuple)):
        return False, f"'x' must be list, got {type(x).__name__}"

    if len(x) != n:
        return False, f"'x' length {len(x)} != n={n}"

    for j, val in enumerate(x):
        if not isinstance(val, (int, float)):
            return False, f"x[{j}] must be numeric, got {type(val).__name__}"
        if val not in (0, 1):
            return False, f"x[{j}]={val} is not binary (must be 0 or 1)"

    tol = 1e-6
    for i in range(m):
        lhs = sum(r[i][j] * x[j] for j in range(n))
        if lhs - b[i] > tol:
            return False, f"constraint {i} violated: consumption {lhs} exceeds limit {b[i]}"

    return True, None
'''
