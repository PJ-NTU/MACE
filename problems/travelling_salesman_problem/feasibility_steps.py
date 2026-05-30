"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    tour = solution.get("tour")
    if tour is None:
        return False, "solution missing 'tour' key"

    if not isinstance(tour, (list, tuple)):
        return False, f"'tour' must be list or tuple, got {type(tour).__name__}"

    num_nodes = len(nodes)

    if len(tour) != num_nodes:
        return False, f"tour length {len(tour)} != expected {num_nodes}"

    if any(not isinstance(x, int) for x in tour):
        return False, "tour entries must be int"

    nodes_set = set(tour)
    if len(nodes_set) != num_nodes:
        return False, f"tour contains {len(nodes_set)} unique nodes, expected {num_nodes} (duplicates present)"

    expected_nodes = set(range(num_nodes))
    if nodes_set != expected_nodes:
        return False, "tour contains out-of-range or missing node indices"

    return True, None
'''
