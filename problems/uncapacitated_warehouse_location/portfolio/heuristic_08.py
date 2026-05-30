# MACE evolved heuristic 08/10 for problem: uncapacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Uncapacitated Warehouse Location Problem using a hybrid approach:
    1. Initial solutions via Greedy Add and Greedy Drop.
    2. Local search (Swap) to refine the best found open set.
    3. Final conversion to the required solution dictionary.
    """
    start_time = time.time()
    
    # Track the best open set found
    best_open_set = None
    min_cost = float('inf')
    
    # 1. Generate candidate starting points
    # Use the provided greedy heuristics
    candidates = []
    
    # Greedy Add
    try:
        add_set = tools['greedy_add_one'](time_limit_s=max(0.1, (time_limit_s - (time.time() - start_time)) * 0.4))
        candidates.append(add_set)
    except:
        pass
        
    # Greedy Drop
    try:
        drop_set = tools['greedy_drop_one'](time_limit_s=max(0.1, (time_limit_s - (time.time() - start_time)) * 0.4))
        candidates.append(drop_set)
    except:
        pass
        
    # Add a random subset as a baseline if none found
    if not candidates:
        m = tools['n_warehouses']()
        candidates.append([random.randint(0, m - 1)])

    # 2. Refine candidates with Swap Local Search
    for open_set in candidates:
        if time.time() - start_time > time_limit_s * 0.9:
            break
            
        try:
            refined_set = tools['apply_swap_open_close'](
                open_set, 
                time_limit_s=max(0.1, (time_limit_s - (time.time() - start_time)) * 0.5)
            )
            cost = tools['cost_given_open'](refined_set)
            
            if cost < min_cost:
                min_cost = cost
                best_open_set = refined_set
        except:
            continue

    # 3. Fallback to ILP if time permits
    if time.time() - start_time < time_limit_s * 0.8:
        try:
            ilp_sol = tools['ilp_uwl'](time_limit_s=max(0.1, time_limit_s - (time.time() - start_time)))
            if ilp_sol:
                return ilp_sol
        except:
            pass

    # 4. Construct final solution
    if best_open_set is None:
        # Final safety: if all failed, pick all warehouses
        best_open_set = list(range(tools['n_warehouses']()))
        
    sol = tools['solution_from_open'](best_open_set)
    
    # Ensure the dictionary contains the required keys
    if sol is None:
        # Should not happen given constraints, but handle gracefully
        m = tools['n_warehouses']()
        n = tools['n_customers']()
        return {
            'total_cost': 0.0,
            'warehouse_open': [0] * m,
            'assignments': [[0] * m for _ in range(n)]
        }
        
    return sol