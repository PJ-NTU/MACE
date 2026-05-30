"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"
    if "path" not in solution:
        return False, "solution missing 'path' key"
    path = solution["path"]
    if not isinstance(path, (list, tuple)):
        return False, f"'path' must be list, got {type(path).__name__}"
    if len(path) == 0:
        return False, "path is empty"
    if path[0] != 1:
        return False, f"path must start at vertex 1, got {path[0]}"
    if path[-1] != n:
        return False, f"path must end at vertex {n}, got {path[-1]}"
    for vertex in path:
        if not isinstance(vertex, int):
            return False, f"path entries must be int, got {type(vertex).__name__}"
        if vertex < 1 or vertex > n:
            return False, f"vertex {vertex} is out of valid range [1, {n}]"
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        arc_found = any(dest == v for (dest, _, _) in graph.get(u, []))
        if not arc_found:
            return False, f"no valid arc from vertex {u} to vertex {v}"
    total_resources = [0.0] * K
    for vertex in path:
        for k in range(K):
            total_resources[k] += vertex_resources[vertex - 1][k]
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        for (dest, arc_cost, arc_res) in graph.get(u, []):
            if dest == v:
                for k in range(K):
                    total_resources[k] += arc_res[k]
                break
    for k in range(K):
        if total_resources[k] < lower_bounds[k] - 1e-6 or total_resources[k] > upper_bounds[k] + 1e-6:
            return False, (f"total consumption for resource {k} is {total_resources[k]}, "
                           f"outside bounds [{lower_bounds[k]}, {upper_bounds[k]}]")
    return True, None
'''
