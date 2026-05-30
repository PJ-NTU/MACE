"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"
    if "selected_columns" not in solution:
        return False, "solution missing 'selected_columns' key"
    selected_columns = solution["selected_columns"]
    if not isinstance(selected_columns, (list, tuple)):
        return False, f"'selected_columns' must be list, got {type(selected_columns).__name__}"
    if any(not isinstance(c, int) for c in selected_columns):
        return False, "'selected_columns' entries must be int"
    selected_set = set(selected_columns)
    for col in selected_set:
        if col < 1 or col > n:
            return False, f"column {col} out of bounds (must be between 1 and {n})"
    uncovered = []
    for i in range(m):
        if not set(row_cover[i]).intersection(selected_set):
            uncovered.append(i + 1)
    if uncovered:
        return False, "rows not covered: {}".format(', '.join(map(str, uncovered[:10])))
    return True, None
'''
