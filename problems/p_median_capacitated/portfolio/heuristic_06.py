# MACE evolved heuristic 06/10 for problem: p_median_capacitated
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized Capacitated P-Median heuristic:
    1. Prioritize ILP for smaller instances or high-quality starts.
    2. Use a hybrid construction: combine greedy seeding with randomized 
       perturbations to explore the solution space effectively.
    3. Iterative Local Search (ILS) with adaptive timing to refine 
       the best-found solutions.
    """
    start_time = time.time()
    n = tools['n_customers']()
    p = tools['p']()
    
    # Reserve time for ILP
    ilp_time = min(time_limit_s * 0.4, 5.0)
    ilp_res = tools['ilp_cpm'](time_limit_s=ilp_time)
    
    best_sol = ilp_res
    best_obj = ilp_res['objective'] if ilp_res else float('inf')
    
    # 2. Iterative Improvement
    # We use a mix of greedy seeding and random sampling to find good initial sets.
    # Then apply swap-based local search.
    
    greedy_seeds = [tools['greedy_p_picks_by_distance']()]
    
    # Main loop for multi-start local search
    while time.time() - start_time < time_limit_s * 0.9:
        # Determine initial set
        if greedy_seeds:
            current_medians = greedy_seeds.pop(0)
        else:
            # Diversification: Random selection
            current_medians = random.sample(range(n), p)
            
        assignments = tools['assignment_by_nearest_feasible'](current_medians)
        
        # If unassignable, try to fix by swapping an open with a random closed
        if -1 in assignments:
            continue
            
        # Refine using the provided local search tool
        remaining = time_limit_s - (time.time() - start_time)
        if remaining < 0.1:
            break
            
        # Limit local search to a slice of remaining time to allow for more restarts
        ls_time = min(remaining * 0.5, 1.0)
        new_medians, new_assignments = tools['apply_swap_open_close'](
            current_medians, t_limit=ls_time
        )
        
        candidate = tools['to_solution'](new_medians, new_assignments)
        
        # Verify feasibility
        is_f, _ = tools['is_feasible'](candidate)
        if is_f:
            obj = candidate['objective']
            if obj < best_obj:
                best_obj = obj
                best_sol = candidate
                
    # Final fallback if no better solution found than ILP
    if best_sol is None:
        # One last attempt with purely greedy if all else fails
        fallback_medians = tools['greedy_p_picks_by_distance']()
        fallback_assign = tools['assignment_by_nearest_feasible'](fallback_medians)
        if -1 not in fallback_assign:
            best_sol = tools['to_solution'](fallback_medians, fallback_assign)
            
    return best_sol if best_sol else (ilp_res or {})