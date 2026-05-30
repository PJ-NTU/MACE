"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # Required keys present
    for key in ("objective", "medians", "assignments"):
        if key not in solution:
            return False, f"solution missing '{key}' key"

    medians = solution["medians"]
    assignments = solution["assignments"]

    # Type checks
    if not isinstance(medians, (list, tuple)):
        return False, f"'medians' must be list, got {type(medians).__name__}"
    if not isinstance(assignments, (list, tuple)):
        return False, f"'assignments' must be list, got {type(assignments).__name__}"

    # Length checks
    if len(medians) != p:
        return False, f"'medians' length {len(medians)} != p={p}"
    if len(assignments) != n:
        return False, f"'assignments' length {len(assignments)} != n={n}"

    # Build customer id set for validity checks
    cust_ids = {cust[0] for cust in customers}

    # Each median must be a valid customer id
    for m in medians:
        if m not in cust_ids:
            return False, f"median {m} is not a valid customer id"

    medians_set = set(medians)

    # Each assignment must be one of the selected medians
    for idx, a in enumerate(assignments):
        if a not in medians_set:
            return False, f"customer {idx + 1} assigned to {a} which is not a selected median"

    # Capacity constraints
    capacity_usage = {m: 0.0 for m in medians_set}
    for i, a in enumerate(assignments):
        demand = customers[i][3]
        capacity_usage[a] += demand
    for m, used in capacity_usage.items():
        if used > Q + 1e-6:
            return False, f"capacity exceeded for median {m}: used {used:.4f} > Q={Q:.4f}"

    return True, None
'''
