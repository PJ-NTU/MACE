"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    # (1) solution dict shape
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # (2) required keys present
    if "structure" not in solution:
        return False, "solution missing 'structure' key"

    structure = solution["structure"]

    # (3) correct type for 'structure'
    if not isinstance(structure, dict):
        return False, f"'structure' must be dict, got {type(structure).__name__}"

    # (4) per-element value constraints
    valid_countries = set(range(1, N + 1))

    for child, parent in structure.items():
        if not isinstance(child, int):
            return False, f"structure key (child) must be int, got {type(child).__name__}"
        if child not in valid_countries:
            return False, f"child {child} out of valid range [1, {N}]"
        if not isinstance(parent, int):
            return False, f"structure value (parent) must be int, got {type(parent).__name__}"
        if parent != 0 and parent not in valid_countries:
            return False, f"parent {parent} out of valid range [0, {N}]"
        if child == parent:
            return False, f"country {child} cannot be its own parent"

    # (5) cross-element / global constraints

    # The target country must appear in the structure with parent 0
    if target not in structure:
        return False, f"target country {target} must be in structure"
    if structure[target] != 0:
        return False, f"target country {target} must have parent 0, got {structure[target]}"

    # No other country should have parent 0
    for child, parent in structure.items():
        if child != target and parent == 0:
            return False, f"only target country may have parent 0, but country {child} also has parent 0"

    # The structure must form a valid tree (no cycles, all nodes reachable from target)
    # Build children map
    children = {i: [] for i in range(1, N + 1)}
    for child, parent in structure.items():
        if parent != 0:
            children[parent].append(child)

    # Check no cycles by traversing from target via BFS/DFS
    visited = set()
    stack = [target]
    while stack:
        node = stack.pop()
        if node in visited:
            return False, f"cycle detected in structure involving country {node}"
        visited.add(node)
        for c in children.get(node, []):
            stack.append(c)

    # Every country listed in structure must be reachable from target
    for child in structure:
        if child not in visited:
            return False, f"country {child} is in structure but not reachable from target {target}"

    return True, None
'''
