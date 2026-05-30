"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    assignments = solution.get("assignments")
    if assignments is None:
        return False, "solution missing 'assignments' key"

    if not isinstance(assignments, (list, tuple)):
        return False, f"'assignments' must be list, got {type(assignments).__name__}"

    if len(assignments) != n:
        return False, f"assignments length {len(assignments)} != n={n}"

    for j, agent in enumerate(assignments):
        if not isinstance(agent, int):
            return False, f"assignments[{j}] must be int, got {type(agent).__name__}"
        if agent < 1 or agent > m:
            return False, f"assignments[{j}]={agent} out of valid range [1, {m}]"

    agent_consumption = [0.0] * m
    for j, agent in enumerate(assignments):
        agent_index = agent - 1
        agent_consumption[agent_index] += consumption_matrix[agent_index][j]

    for i in range(m):
        if agent_consumption[i] > capacities[i]:
            return False, (
                f"capacity constraint violated for agent {i + 1}: "
                f"consumption {agent_consumption[i]} exceeds capacity {capacities[i]}"
            )

    return True, None
'''
