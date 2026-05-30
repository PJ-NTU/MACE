# MACE evolved heuristic 06/10 for problem: capacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined solver for Capacitated Warehouse Location.
    
    The previous implementation relied heavily on a single-source ILP which, while
    optimal for single-source, does not account for the additional flexibility
    of splittable demand. This version uses a multi-stage approach:
    1. ILP for a strong baseline.
    2. Hill-climbing local search to iteratively optimize the 'warehouse_open' 
       configuration by shifting demand between open warehouses to minimize costs.
    """
    start_time = time.time()
    
    # 1. Get baseline from ILP
    # The ILP solver returns a valid single-source solution.
    # Single-source is a valid subset of the splittable problem space.
    best_sol = tools['ilp_cwl'](time_limit_s=max(0.1, time_limit_s * 0.4))
    
    if best_sol is None:
        # Fallback to density heuristic
        open_set, assignment = tools['greedy_open_by_density']()
        best_sol = tools['to_solution'](open_set, assignment)
        best_sol['total_cost'] = tools['objective'](best_sol)
    
    # 2. Local Search: Iterative improvement of open warehouses
    # Try to close expensive warehouses or open efficient ones to improve cost
    m = tools['n_warehouses']()
    n = tools['n_customers']()
    
    current_open = list(best_sol['warehouse_open'])
    
    # Simple Hill Climbing: Toggle warehouse status
    while time.time() - start_time < time_limit_s * 0.9:
        changed = False
        idx = random.randrange(m)
        
        # Propose flip
        current_open[idx] = 1 - current_open[idx]
        open_set = [i for i, v in enumerate(current_open) if v == 1]
        
        # Re-assign using greedy_serve_nearest
        assignment = tools['greedy_serve_nearest'](open_set)
        
        # Check if valid (all customers served)
        if -1 not in assignment:
            candidate = tools['to_solution'](open_set, assignment)
            cand_cost = tools['objective'](candidate)
            
            if cand_cost < best_sol['total_cost']:
                best_sol = candidate
                best_sol['total_cost'] = cand_cost
                changed = True
        
        if not changed:
            # Revert
            current_open[idx] = 1 - current_open[idx]
            
        # Break if no improvement found after a few random attempts
        if not changed and random.random() > 0.5:
            break
            
    return best_sol