"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    # (1) solution dict shape
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # (2) required keys present
    selected_columns = solution.get("selected_columns")
    if selected_columns is None:
        return False, "solution missing 'selected_columns' key"

    # (3) correct type
    if not isinstance(selected_columns, list):
        return False, f"'selected_columns' must be list, got {type(selected_columns).__name__}"

    # (4) per-element value constraints
    for col in selected_columns:
        if not isinstance(col, int):
            return False, f"column index {col!r} must be int"
        if col < 1 or col > num_columns:
            return False, f"column index {col} out of range [1, {num_columns}]"

    # strictly increasing order and no duplicates
    if selected_columns != sorted(selected_columns) or len(selected_columns) != len(set(selected_columns)):
        return False, "selected_columns must be in strictly increasing order with no duplicates"

    # (5) cross-element / global constraints
    for col in selected_columns:
        if col not in columns_info:
            return False, f"column {col} not found in columns_info"
        _, covered_rows = columns_info[col]
        for r in covered_rows:
            if r < 1 or r > num_rows:
                return False, f"row index {r} in column {col} out of range [1, {num_rows}]"

    row_coverage = [0] * (num_rows + 1)
    for col in selected_columns:
        _, covered_rows = columns_info[col]
        for r in covered_rows:
            row_coverage[r] += 1

    for r in range(1, num_rows + 1):
        if row_coverage[r] != 1:
            return False, f"row {r} is covered {row_coverage[r]} times; must be covered exactly once"

    return True, None
'''
