"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    # (1) solution dict shape
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # (2) normalize keys to integers and check all vertices 1..n are present
    normalized = {}
    for key, value in solution.items():
        try:
            vertex = int(key)
        except (ValueError, TypeError):
            return False, f"solution key {key!r} cannot be converted to int"
        normalized[vertex] = value

    expected_vertices = set(range(1, n + 1))
    if set(normalized.keys()) != expected_vertices:
        missing = expected_vertices - set(normalized.keys())
        extra = set(normalized.keys()) - expected_vertices
        msg = []
        if missing:
            msg.append(f"missing vertices: {sorted(missing)}")
        if extra:
            msg.append(f"extra vertices: {sorted(extra)}")
        return False, "solution must assign a color to every vertex 1..n; " + "; ".join(msg)

    # (3) & (4) each color must be a positive integer
    for v, color in normalized.items():
        if not isinstance(color, int):
            return False, f"color for vertex {v} must be int, got {type(color).__name__}"
        if color < 1:
            return False, f"color for vertex {v} must be >= 1, got {color}"

    # (5) no two adjacent vertices may share the same color
    for u in range(1, n + 1):
        for v in adjacency[u]:
            if u < v:
                if normalized[u] == normalized[v]:
                    return False, f"conflict: adjacent vertices {u} and {v} share color {normalized[u]}"

    return True, None
'''
