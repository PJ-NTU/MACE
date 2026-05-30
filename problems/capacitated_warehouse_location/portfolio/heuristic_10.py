# MACE evolved heuristic 10/10 for problem: capacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined heuristic combining ILP for optimality where possible,
    and a robust Local Search (Swap/Open/Close) for larger instances.
    Modified to improve robustness on instances where greedy_open_by_density 
    fails to find an initial feasible solution by trying alternate initializations.
    """
    start_time = time.time()
    m = tools['n_warehouses']()
    n = tools['n_customers']()

    # 1. Attempt ILP for optimal core
    ilp_result = tools['ilp_cwl'](time_limit_s=max(0.5, time_limit_s * 0.7))
    if ilp_result is not None:
        return ilp_result

    # 2. Construction: Start with a dense-based heuristic
    open_set, assignment = tools['greedy_open_by_density']()
    
    # MODIFICATION: If greedy_open_by_density fails, try opening all warehouses 
    # to guarantee feasibility before entering local search.
    unassigned = tools['unassigned_customers'](assignment)
    if unassigned:
        open_set = list(range(m))
        assignment = tools['greedy_serve_nearest'](open_set)
    
    best_sol = tools['to_solution'](open_set, assignment)
    best_cost = best_sol['total_cost']
    
    # 3. Local Search: Refine configuration using swap/open/close
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.2:
        new_open, new_assign = tools['apply_swap_open_close'](
            open_set, 
            time_limit_s=min(remaining_time, 2.0)
        )
        
        # Validate and update
        if -1 not in new_assign:
            candidate_sol = tools['to_solution'](new_open, new_assign)
            if candidate_sol['total_cost'] < best_cost:
                best_sol = candidate_sol
                best_cost = candidate_sol['total_cost']
                open_set = new_open
                assignment = new_assign

    # 4. Final safety check: If still somehow infeasible, force open all
    if -1 in assignment:
        open_set = list(range(m))
        assignment = tools['greedy_serve_nearest'](open_set)
        return tools['to_solution'](open_set, assignment)

    return best_sol