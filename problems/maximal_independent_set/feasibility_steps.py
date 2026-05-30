"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    mis_nodes = solution.get("mis_nodes")
    if mis_nodes is None:
        return False, "solution missing 'mis_nodes' key"

    if not isinstance(mis_nodes, list):
        return False, f"'mis_nodes' must be list, got {type(mis_nodes).__name__}"

    node_set = set(graph.nodes())
    for node in mis_nodes:
        if node not in node_set:
            return False, f"node {node} in solution does not exist in graph"

    if len(mis_nodes) != len(set(mis_nodes)):
        return False, "duplicate nodes in mis_nodes"

    for i in range(len(mis_nodes)):
        for j in range(i + 1, len(mis_nodes)):
            if graph.has_edge(mis_nodes[i], mis_nodes[j]):
                return False, f"not an independent set: edge exists between {mis_nodes[i]} and {mis_nodes[j]}"

    return True, None
'''
