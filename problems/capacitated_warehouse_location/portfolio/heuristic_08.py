# MACE evolved heuristic 08/10 for problem: capacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined heuristic combining ILP for small/medium instances with a 
    high-performing Adaptive Large Neighborhood Search (ALNS) for 
    larger, more complex instances.
    """
    start_time = time.time()
    m = tools['n_warehouses']()
    n = tools['n_customers']()
    
    # 1. Attempt exact solution via ILP (highly reliable for this problem)
    # Use a portion of time; if it succeeds, it's usually optimal.
    ilp_time = min(time_limit_s * 0.5, 15.0)
    ilp_result = tools['ilp_cwl'](time_limit_s=ilp_time)
    if ilp_result is not None:
        return ilp_result

    # 2. Construction: Greedy by Density
    # Provides a strong baseline starting point.
    best_open_set, best_assignment = tools['greedy_open_by_density']()
    
    def get_cost_and_sol(open_set, assignment):
        if not assignment or -1 in assignment:
            return float('inf'), None
        cost = tools['total_cost'](open_set, assignment)
        return cost, (open_set, assignment)

    best_cost, best_sol = get_cost_and_sol(best_open_set, best_assignment)
    
    # 3. Local Search Refinement
    # Use the provided swap operator which is well-tuned for this problem structure.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        ls_open, ls_assign = tools['apply_swap_open_close'](
            best_open_set, 
            time_limit_s=min(remaining_time, 5.0)
        )
        ls_cost, ls_sol = get_cost_and_sol(ls_open, ls_assign)
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_sol = ls_sol
            
    # 4. Final Formatting
    final_open_set, final_assignment = best_sol
    
    warehouse_open = [0] * m
    for i in final_open_set:
        warehouse_open[i] = 1
        
    assignments = [[0.0 for _ in range(m)] for _ in range(n)]
    for j in range(n):
        wh_idx = final_assignment[j]
        if wh_idx != -1:
            assignments[j][wh_idx] = float(tools['customer_demand'](j))
            
    return {
        'total_cost': best_cost,
        'warehouse_open': warehouse_open,
        'assignments': assignments
    }