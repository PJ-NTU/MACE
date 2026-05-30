# MACE evolved heuristic 01/10 for problem: uncapacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Uncapacitated Warehouse Location Problem using a hybrid strategy:
    1. Initial solutions via Greedy ADD and Greedy DROP heuristics.
    2. Local search improvement using swap operations.
    3. Final refinement via ILP if time permits.
    """
    start_time = time.time()
    
    def get_remaining_time():
        return max(0.1, time_limit_s - (time.time() - start_time))

    best_open_set = None
    best_cost = float('inf')

    # 1. Generate candidate sets using greedy heuristics
    candidates = []
    try:
        candidates.append(tools['greedy_add_one'](time_limit_s=get_remaining_time()))
        candidates.append(tools['greedy_drop_one'](time_limit_s=get_remaining_time()))
    except Exception:
        pass

    # 2. Local search: improve candidates via swap
    for open_set in candidates:
        if not open_set:
            continue
        
        improved_set = tools['apply_swap_open_close'](open_set, time_limit_s=get_remaining_time())
        cost = tools['cost_given_open'](improved_set)
        
        if cost < best_cost:
            best_cost = cost
            best_open_set = improved_set

    # 3. ILP Refinement
    # If we have significant time, try to solve via ILP.
    # We use the best found heuristic solution as a warm start if the ILP tool supports it
    # or just use the tool to find a better global optimum.
    if get_remaining_time() > 1.0:
        try:
            ilp_sol = tools['ilp_uwl'](time_limit_s=get_remaining_time())
            if ilp_sol:
                ilp_cost = ilp_sol.get('total_cost', float('inf'))
                if ilp_cost < best_cost:
                    return ilp_sol
        except Exception:
            pass

    # Fallback to the best heuristic solution found
    if best_open_set is None:
        # Emergency: Open all, then try to find one valid
        m = tools['n_warehouses']()
        best_open_set = list(range(m))

    return tools['solution_from_open'](best_open_set)