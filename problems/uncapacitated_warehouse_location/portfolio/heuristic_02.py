# MACE evolved heuristic 02/10 for problem: uncapacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Uncapacitated Warehouse Location Problem using a hybrid approach:
    1. Initial solutions via Greedy ADD and Greedy DROP heuristics.
    2. Local search refinement via Swap and Variable Neighborhood Descent (VND).
    3. ILP solver as a fallback/refinement if time allows.
    """
    start_time = time.time()
    
    def get_time_left():
        return time_limit_s - (time.time() - start_time)

    best_solution = None
    best_cost = float('inf')

    # 1. Generate candidate start sets from greedy heuristics
    candidates = []
    
    # Greedy ADD
    if get_time_left() > 0.1:
        add_set = tools['greedy_add_one'](time_limit_s=max(0.01, get_time_left() * 0.2))
        candidates.append(add_set)
    
    # Greedy DROP
    if get_time_left() > 0.1:
        drop_set = tools['greedy_drop_one'](time_limit_s=max(0.01, get_time_left() * 0.2))
        candidates.append(drop_set)

    # 2. Refine candidates with Swap local search
    for open_set in candidates:
        if get_time_left() < 0.1:
            break
        
        refined_set = tools['apply_swap_open_close'](open_set, time_limit_s=max(0.01, get_time_left() * 0.3))
        
        # Evaluate
        cost = tools['cost_given_open'](refined_set)
        if cost < best_cost:
            best_cost = cost
            best_solution = tools['solution_from_open'](refined_set)

    # 3. Use ILP solver if time permits for final polish
    if get_time_left() > 0.5:
        # Use our best found solution as a guide if possible, 
        # but the tool interface allows direct ILP solving.
        ilp_sol = tools['ilp_uwl'](time_limit_s=min(5.0, get_time_left() * 0.8))
        if ilp_sol:
            ilp_cost = ilp_sol.get('total_cost', float('inf'))
            if ilp_cost < best_cost:
                best_solution = ilp_sol
                best_cost = ilp_cost

    # 4. Final safety check
    if best_solution is None:
        # Fallback to a simple heuristic if nothing else worked
        fallback_set = tools['greedy_add_one'](time_limit_s=0.1)
        best_solution = tools['solution_from_open'](fallback_set)
        
    return best_solution