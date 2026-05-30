"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    if "assignment" not in solution:
        return False, "solution missing 'assignment' key"

    assignment = solution["assignment"]

    if not isinstance(assignment, (list, tuple)):
        return False, f"'assignment' must be list, got {type(assignment).__name__}"

    n_individuals = len(data)
    if len(assignment) != n_individuals:
        return False, f"assignment length {len(assignment)} != n_individuals={n_individuals}"

    for idx, g in enumerate(assignment, start=1):
        if not isinstance(g, int) or g < 1:
            return False, f"invalid group assignment at position {idx}: {g}. Must be a positive integer"

    groups = set(assignment)
    if len(groups) != 8:
        return False, f"expected exactly 8 distinct groups, got {len(groups)}"

    return True, None
'''
