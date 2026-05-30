"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    if "warehouse_open" not in solution:
        return False, "solution missing 'warehouse_open' key"
    if "assignments" not in solution:
        return False, "solution missing 'assignments' key"

    warehouse_open = solution["warehouse_open"]
    assignments = solution["assignments"]

    if not isinstance(warehouse_open, (list, tuple)):
        return False, f"'warehouse_open' must be list, got {type(warehouse_open).__name__}"
    if len(warehouse_open) != m:
        return False, f"'warehouse_open' length {len(warehouse_open)} != m={m}"
    if any(v not in (0, 1) for v in warehouse_open):
        return False, "'warehouse_open' entries must be 0 or 1"

    if not isinstance(assignments, (list, tuple)):
        return False, f"'assignments' must be list, got {type(assignments).__name__}"
    if len(assignments) != n:
        return False, f"'assignments' length {len(assignments)} != n={n}"
    for j, row in enumerate(assignments):
        if not isinstance(row, (list, tuple)):
            return False, f"assignments[{j}] must be list, got {type(row).__name__}"
        if len(row) != m:
            return False, f"assignments[{j}] length {len(row)} != m={m}"

    for j in range(n):
        customer_demand = customers[j]['demand']
        allocated_amount = sum(assignments[j])
        if abs(allocated_amount - customer_demand) > 1e-6:
            return False, (f"Customer {j} demand violation: total assigned {allocated_amount} "
                           f"!= demand {customer_demand}")
        for i in range(m):
            allocation = assignments[j][i]
            if allocation < 0:
                return False, f"Customer {j} has negative allocation {allocation} for warehouse {i}"
            if allocation > 0 and warehouse_open[i] != 1:
                return False, (f"Customer {j} has allocation {allocation} for warehouse {i}, "
                               f"which is closed")

    assigned_demand = [0.0] * m
    for i in range(m):
        for j in range(n):
            assigned_demand[i] += assignments[j][i]
    for i in range(m):
        if assigned_demand[i] > warehouses[i]['capacity'] + 1e-6:
            excess = assigned_demand[i] - warehouses[i]['capacity']
            return False, f"Warehouse {i} exceeds its capacity by {excess} units"

    return True, None
'''
