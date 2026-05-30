# MACE evolved heuristic 03/10 for problem: capacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Heuristic for Capacitated Warehouse Location:
    1. Generates a warm start using greedy_open_by_density.
    2. Refines the solution using Local Search (swap/open/close).
    3. Converts the single-source assignment to the splittable format required.
    """
    start_time = time.time()
    
    # 1. Initial Construction
    # We use the provided greedy heuristic to get a baseline feasible solution
    open_set, assignment = tools['greedy_open_by_density']()
    
    # 2. Local Search
    # Refine the open_set and assignment using the provided local search tool
    # We allocate ~80% of the time to local search
    ls_time_limit = max(0.1, (time_limit_s - (time.time() - start_time)) * 0.8)
    
    # The tool returns (open_set, assignment)
    refined_open_set, refined_assignment = tools['apply_swap_open_close'](
        open_set=open_set, 
        time_limit_s=ls_time_limit
    )
    
    # 3. Format conversion
    # The problem asks for splittable assignments (n x m).
    # Since our LS tools work on single-source assignments, we convert 
    # the single-source assignment to the full matrix format.
    m = tools['n_warehouses']()
    n = tools['n_customers']()
    
    warehouse_open = [0] * m
    for idx in refined_open_set:
        warehouse_open[idx] = 1
        
    assignments = [[0.0 for _ in range(m)] for _ in range(n)]
    for j in range(n):
        wh_idx = refined_assignment[j]
        if wh_idx != -1:
            assignments[j][wh_idx] = tools['customer_demand'](j)
            
    # 4. Final verification and return
    # Use the tools to compute the objective and ensure the result is valid
    solution = {
        'warehouse_open': warehouse_open,
        'assignments': assignments
    }
    
    # Check feasibility via the provided tool before returning
    is_feasible, msg = tools['is_feasible'](solution)
    
    if is_feasible:
        cost = tools['objective'](solution)
        solution['total_cost'] = cost
        return solution
    else:
        # Fallback: if local search failed, try the ILP solver as a last resort
        # if time permits.
        remaining = time_limit_s - (time.time() - start_time)
        if remaining > 0.5:
            res = tools['ilp_cwl'](time_limit_s=remaining)
            if res:
                return res
        
        # If all else fails, return the construction heuristic result
        # Note: The problem expects a valid dictionary. 
        # If the construction heuristic is feasible, return it.
        sol_init = tools['to_solution'](open_set, assignment)
        sol_init['total_cost'] = tools['objective'](sol_init)
        return sol_init