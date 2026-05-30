# MACE evolved heuristic 01/10 for problem: capacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Implements a hybrid approach:
    1. Uses the ILP solver as the primary engine for optimal single-source assignments.
    2. Falls back to a Greedy + Local Search (Swap Open/Close) if ILP fails or times out.
    """
    start_time = time.time()
    
    # 1. Try ILP approach first
    ilp_result = tools['ilp_cwl'](time_limit_s=max(0.1, time_limit_s * 0.8))
    if ilp_result is not None:
        # ILP provides a single-source solution (assignments[j][i] is demand or 0)
        # which is a valid subset of the splittable demand problem.
        return ilp_result

    # 2. Fallback: Greedy Construction
    open_set, assignment = tools['greedy_open_by_density']()
    
    # Check if we have a valid assignment
    unassigned = tools['unassigned_customers'](assignment)
    if unassigned:
        # Attempt to brute-force open all warehouses to solve unassigned
        open_set = list(range(tools['n_warehouses']()))
        assignment = tools['greedy_serve_nearest'](open_set)
    
    # 3. Refine with Local Search
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        open_set_list, assignment = tools['apply_swap_open_close'](
            open_set, 
            time_limit_s=min(remaining_time, 2.0)
        )
    
    # 4. Final conversion to required format
    # The tools return single-source, but the problem allows splittable.
    # A single-source solution is a valid splittable solution.
    # Convert internal representation to the required dict format.
    m = tools['n_warehouses']()
    n = tools['n_customers']()
    
    warehouse_open = [0] * m
    for i in open_set_list:
        warehouse_open[i] = 1
        
    assignments = [[0.0 for _ in range(m)] for _ in range(n)]
    for j in range(n):
        wh_idx = assignment[j]
        if wh_idx != -1:
            assignments[j][wh_idx] = tools['customer_demand'](j)
            
    # Calculate objective
    # Note: tools['total_cost'] assumes single-source, which matches our logic.
    cost = tools['total_cost'](open_set_list, assignment)
    
    return {
        'total_cost': cost,
        'warehouse_open': warehouse_open,
        'assignments': assignments
    }