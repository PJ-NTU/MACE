# MACE evolved heuristic 07/10 for problem: uncapacitated_warehouse_location
import time

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve the Uncapacitated Warehouse Location Problem using a hybrid approach:
    1. Greedy Add/Drop heuristics for initial promising sets.
    2. Local search (Swap) to refine the solution.
    3. ILP solver for final refinement if time permits.
    """
    start_time = time.time()
    
    # 1. Generate initial candidates using greedy heuristics
    candidates = []
    
    # Greedy Add
    try:
        set_add = tools['greedy_add_one'](time_limit_s=max(0.1, (time_limit_s - (time.time() - start_time)) / 3))
        candidates.append(set_add)
    except:
        pass
        
    # Greedy Drop
    try:
        set_drop = tools['greedy_drop_one'](time_limit_s=max(0.1, (time_limit_s - (time.time() - start_time)) / 3))
        candidates.append(set_drop)
    except:
        pass
    
    # Evaluate and refine candidates via Swap
    best_cost = float('inf')
    best_open_set = None
    
    for open_set in candidates:
        if time.time() - start_time > time_limit_s * 0.8:
            break
            
        refined_set = tools['apply_swap_open_close'](open_set, time_limit_s=max(0.1, (time_limit_s - (time.time() - start_time)) / 2))
        cost = tools['cost_given_open'](refined_set)
        
        if cost < best_cost:
            best_cost = cost
            best_open_set = refined_set
            
    # 2. Final refinement using ILP if time allows
    if time.time() - start_time < time_limit_s * 0.9:
        try:
            # Use ILP to search for a better solution in the remaining time
            ilp_sol = tools['ilp_uwl'](time_limit_s=max(0.1, time_limit_s - (time.time() - start_time)))
            if ilp_sol:
                ilp_cost = ilp_sol.get('total_cost', float('inf'))
                if ilp_cost < best_cost:
                    return ilp_sol
        except:
            pass
            
    # 3. Construct final solution from best_open_set
    if best_open_set is None:
        # Fallback: Open all warehouses if everything failed
        best_open_set = list(range(tools['n_warehouses']()))
        
    final_sol = tools['solution_from_open'](best_open_set)
    
    # Ensure it's a valid dict
    if final_sol:
        return final_sol
    
    # Ultimate emergency fallback
    m = tools['n_warehouses']()
    n = tools['n_customers']()
    return {
        'total_cost': float('inf'),
        'warehouse_open': [1] * m,
        'assignments': [[1 if i == 0 else 0 for i in range(m)] for j in range(n)]
    }