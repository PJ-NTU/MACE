# MACE evolved heuristic 04/10 for problem: p_median_capacitated
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
    
    # 1. Try ILP solver first. Modified to reserve more time for the ILP solver 
    # to find a global optimum on smaller instances.
    ilp_res = tools['ilp_cpm'](time_limit_s=max(2.0, time_limit_s * 0.7))
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
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        current_medians, current_assignments = tools['apply_swap_open_close'](
            current_medians, t_limit=remaining_time
        )
    else:
        current_assignments = tools['assignment_by_nearest_feasible'](current_medians)

    # Convert to standard format
    sol = tools['to_solution'](current_medians, current_assignments)
    
    # Check feasibility and return
    is_f, _ = tools['is_feasible'](sol)
    if is_f:
        best_sol = sol
        best_obj = sol['objective']
    
    # If initial failed or poor, try a randomized restart if time permits
    # Changed loop condition to allow more aggressive exploration of the search space
    while time.time() - start_time < time_limit_s * 0.95:
        # Weighted selection: prioritize facilities with lower aggregate cost
        # (Heuristic: select p indices based on a mix of random and greedy)
        all_indices = list(range(n))
        random.shuffle(all_indices)
        rand_medians = all_indices[:p]
        
        assign = tools['assignment_by_nearest_feasible'](rand_medians)
        
        if -1 not in assign:
            candidate = tools['to_solution'](rand_medians, assign)
            obj = candidate['objective']
            if obj < best_obj:
                best_obj = obj
                best_sol = candidate
    
    return best_sol if best_sol else sol