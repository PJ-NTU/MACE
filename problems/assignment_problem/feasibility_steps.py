"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    if "assignment" not in solution:
        return False, "solution missing 'assignment' key"

    assignment_list = solution["assignment"]
    if not isinstance(assignment_list, list):
        return False, f"'assignment' must be list, got {type(assignment_list).__name__}"

    import math

    seen_items = {}
    seen_agents = set()

    for idx, pair in enumerate(assignment_list, start=1):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return False, f"assignment entry {idx} must be a tuple/list of two integers (i, j)"
        i_val, j_val = pair
        if i_val in seen_items:
            return False, f"duplicate assignment for item {i_val}"
        if j_val in seen_agents:
            return False, f"agent {j_val} assigned more than once"
        if not (1 <= i_val <= n and 1 <= j_val <= n):
            return False, f"assignment indices ({i_val}, {j_val}) out of range [1, {n}]"
        seen_items[i_val] = j_val
        seen_agents.add(j_val)

    if len(seen_items) != n:
        return False, f"incomplete assignment: expected {n} assignments, got {len(seen_items)}"

    for i in range(1, n + 1):
        j_val = seen_items[i]
        if cost_matrix[i - 1][j_val - 1] == math.inf:
            return False, f"assignment ({i}, {j_val}) has infinite cost, hence invalid"

    return True, None
'''
