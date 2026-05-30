# MACE evolved heuristic 06/10 for problem: uncapacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Uncapacitated Warehouse Location Problem using a combination
    of greedy heuristics and local search refinement.
    """
    start_time = time.time()
    
    def get_time_left():
        return time_limit_s - (time.time() - start_time)

    # 1. Generate candidate 'open_sets' using greedy strategies
    candidates = []
    
    # Try greedy ADD
    if get_time_left() > 0.1:
        add_set = tools['greedy_add_one'](time_limit_s=max(0.01, get_time_left() * 0.3))
        if add_set:
            candidates.append(add_set)
            
    # Try greedy DROP
    if get_time_left() > 0.1:
        drop_set = tools['greedy_drop_one'](time_limit_s=max(0.01, get_time_left() * 0.3))
        if drop_set:
            candidates.append(drop_set)
            
    # Include a random candidate to diversify
    m = tools['n_warehouses']()
    if m > 0:
        random_k = random.randint(1, max(1, m // 2))
        random_set = sorted(random.sample(range(m), random_k))
        candidates.append(random_set)

    # 2. Refine candidates using Swap local search
    best_cost = float('inf')
    best_open_set = None
    
    for open_set in candidates:
        if get_time_left() < 0.1:
            break
            
        refined_set = tools['apply_swap_open_close'](open_set, time_limit_s=max(0.01, get_time_left() * 0.5))
        cost = tools['cost_given_open'](refined_set)
        
        if cost < best_cost:
            best_cost = cost
            best_open_set = refined_set

    # 3. Last-ditch effort: ILP if time permits
    if get_time_left() > 1.0:
        ilp_sol = tools['ilp_uwl'](time_limit_s=get_time_left() - 0.2)
        if ilp_sol and ilp_sol.get('total_cost', float('inf')) < best_cost:
            return ilp_sol

    # 4. Construct final solution
    if best_open_set is None:
        # Fallback to opening all if something went wrong
        best_open_set = list(range(m))
        
    solution = tools['solution_from_open'](best_open_set)
    
    # Ensure it's returned as a valid dict
    if solution is None:
        # Should not happen given problem constraints
        return {}
        
    return solution