# MACE evolved heuristic 05/10 for problem: uncapacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve the Uncapacitated Warehouse Location Problem using a combination of
    greedy heuristics and local search refinement.
    """
    start_time = time.time()
    
    # 1. Generate initial candidate sets using greedy approaches
    # Greedy add and greedy drop are excellent baselines for UWL.
    candidates = []
    
    # Add heuristic
    try:
        set_add = tools['greedy_add_one'](time_limit_s=max(0.1, (time_limit_s - (time.time() - start_time)) / 3))
        candidates.append(set_add)
    except Exception:
        pass
        
    # Drop heuristic
    try:
        set_drop = tools['greedy_drop_one'](time_limit_s=max(0.1, (time_limit_s - (time.time() - start_time)) / 3))
        candidates.append(set_drop)
    except Exception:
        pass
        
    # 2. Local search refinement (Swap)
    best_open_set = None
    best_cost = float('inf')
    
    for open_set in candidates:
        if not open_set:
            continue
            
        # Refine with swap local search
        remaining_time = max(0.1, (time_limit_s - (time.time() - start_time)))
        refined_set = tools['apply_swap_open_close'](open_set, time_limit_s=remaining_time)
        
        current_cost = tools['cost_given_open'](refined_set)
        if current_cost < best_cost:
            best_cost = current_cost
            best_open_set = refined_set
            
    # 3. Fallback: If heuristics failed, try a random selection or single best
    if best_open_set is None:
        best_open_set = [0] # Default to opening the first one
        
    # 4. Construct final solution
    # The solution_from_open tool returns a dict with 'total_cost', 'warehouse_open', 'assignments'
    sol = tools['solution_from_open'](best_open_set)
    
    # Final sanity check: ensure we return a valid structure
    if sol is None:
        # Emergency recovery: open all warehouses
        m = tools['n_warehouses']()
        all_open = list(range(m))
        sol = tools['solution_from_open'](all_open)
        
    return sol