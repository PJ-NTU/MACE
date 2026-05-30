# MACE evolved heuristic 03/10 for problem: uncapacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve the Uncapacitated Warehouse Location Problem using a combination of
    greedy heuristics and local search refinement.
    """
    start_time = time.time()
    
    # 1. Generate candidate 'open_sets' using greedy heuristics
    # The greedy_add_one and greedy_drop_one are efficient starting points.
    candidates = []
    
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        candidates.append(tools['greedy_add_one'](time_limit_s=remaining_time * 0.3))
    
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        candidates.append(tools['greedy_drop_one'](time_limit_s=remaining_time * 0.3))
        
    # 2. Refine candidates with swap local search
    best_open_set = None
    best_cost = float('inf')
    
    for open_set in candidates:
        remaining_time = time_limit_s - (time.time() - start_time)
        if remaining_time < 0.1:
            break
            
        refined_set = tools['apply_swap_open_close'](open_set, time_limit_s=remaining_time * 0.5)
        current_cost = tools['cost_given_open'](refined_set)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_open_set = refined_set
            
    # 3. Fallback to ILP if time permits
    remaining_time = time_limit_s - (time.time() - start_time)
    if (best_open_set is None or remaining_time > 2.0):
        # Even if we have a solution, ILP might find a better one if time allows
        ilp_sol = tools['ilp_uwl'](time_limit_s=max(0.5, remaining_time - 0.5))
        if ilp_sol:
            return ilp_sol

    # 4. Final construction
    if best_open_set is None:
        # Emergency recovery: open everything if nothing else worked
        m = tools['n_warehouses']()
        best_open_set = list(range(m))
        
    return tools['solution_from_open'](best_open_set)