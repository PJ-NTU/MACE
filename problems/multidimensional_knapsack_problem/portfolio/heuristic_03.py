# MACE evolved heuristic 03/10 for problem: multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Multidimensional Knapsack Problem solver using a GRASP-inspired 
    approach combined with local search refinement.
    """
    start_time = time.time()
    n = instance['n']
    
    # 1. Generate multiple starting points using different greedy strategies
    # to provide a diverse set of feasible solutions.
    candidates = []
    
    # Greedy by profit density
    try:
        sol1 = tools['greedy_by_profit_density']()
        candidates.append(sol1)
    except:
        pass
        
    # Greedy by efficiency
    try:
        sol2 = tools['greedy_by_efficiency']()
        candidates.append(sol2)
    except:
        pass

    # 2. Refine candidates with local search
    best_profit = -1.0
    best_selection = []
    
    # Allocate time for local search
    remaining_time = time_limit_s - (time.time() - start_time)
    if not candidates:
        candidates.append([])
        
    # Use a per-candidate time budget
    ls_time_limit = max(0.1, remaining_time / len(candidates))
    
    for seed_selection in candidates:
        if time.time() - start_time > time_limit_s:
            break
            
        try:
            # Improvement phase
            refined = tools['apply_local_swap_in_out'](seed_selection, time_limit_s=ls_time_limit)
            current_profit = tools['profit_of_selection'](refined)
            
            if current_profit > best_profit:
                best_profit = current_profit
                best_selection = refined
        except:
            continue

    # 3. Final ILP refinement if time permits
    # Use the best greedy/local-search solution as a baseline for the ILP
    if time.time() - start_time < time_limit_s * 0.8:
        try:
            # We fix the current best as a lower bound by allowing the ILP 
            # to search around it or optimize globally.
            ilp_time = time_limit_s - (time.time() - start_time)
            ilp_result = tools['ilp_solve_mkp'](time_limit_s=max(1.0, ilp_time))
            if ilp_result is not None:
                if tools['profit_of_selection'](ilp_result) > best_profit:
                    best_selection = ilp_result
        except:
            pass

    # Convert selection list to binary decision array
    x = [0] * n
    for idx in best_selection:
        if 0 <= idx < n:
            x[idx] = 1
            
    return {'x': x}