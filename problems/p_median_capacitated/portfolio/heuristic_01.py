# MACE evolved heuristic 01/10 for problem: p_median_capacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Capacitated P-Median heuristic:
    1. Start with a greedy seed using facility-customer distance.
    2. Improve via iterative Local Search (swap open/closed facilities).
    3. Use the provided ILP solver if time allows, otherwise return the best found.
    """
    start_time = time.time()
    
    # 1. Try ILP solver first if it's likely to perform well or time is ample
    # The ILP solver is often the strongest performer for this problem type.
    ilp_res = tools['ilp_cpm'](time_limit_s=max(1.0, time_limit_s * 0.5))
    if ilp_res:
        return ilp_res

    # 2. Fallback: Greedy + Local Search
    p = tools['p']()
    n = tools['n_customers']()
    
    best_sol = None
    best_obj = float('inf')

    # Initial construction: Seed based on greedy distance
    current_medians = tools['greedy_p_picks_by_distance']()
    
    # Refine with local search (facility swaps)
    # Give it half of the remaining time
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        current_medians, current_assignments = tools['apply_swap_open_close'](
            current_medians, t_limit=remaining_time
        )
    else:
        # Fallback to simple greedy assignment if swap not possible
        current_assignments = tools['assignment_by_nearest_feasible'](current_medians)

    # Convert to standard format
    sol = tools['to_solution'](current_medians, current_assignments)
    
    # Check feasibility and return
    is_f, _ = tools['is_feasible'](sol)
    if is_f:
        return sol
    
    # If initial failed, try a randomized restart if time permits
    while time.time() - start_time < time_limit_s * 0.8:
        # Randomly pick a different set of p medians
        all_indices = list(range(n))
        random.shuffle(all_indices)
        rand_medians = all_indices[:p]
        
        assign = tools['assignment_by_nearest_feasible'](rand_medians)
        # Check if all customers assigned
        if -1 not in assign:
            candidate = tools['to_solution'](rand_medians, assign)
            obj = candidate['objective']
            if obj < best_obj:
                best_obj = obj
                best_sol = candidate
    
    return best_sol if best_sol else sol