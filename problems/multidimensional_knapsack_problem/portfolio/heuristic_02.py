# MACE evolved heuristic 02/10 for problem: multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Multidimensional Knapsack Problem using a combination of 
    diverse greedy strategies, local search, and randomized hill climbing.
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # 1. Generate candidate warm-start solutions using greedy approaches
    candidates = []
    
    # Greedy by profit density (tightest dimension)
    selection1 = tools['greedy_by_profit_density']()
    candidates.append(selection1)
    
    # Greedy by total efficiency
    selection2 = tools['greedy_by_efficiency']()
    candidates.append(selection2)
    
    # Random selection (if time permits)
    if time_limit_s > 0.5:
        items = list(range(n))
        random.shuffle(items)
        rand_sel = []
        for item in items:
            rand_sel.append(item)
            if not tools['is_within_all_capacities'](rand_sel):
                rand_sel.pop()
        candidates.append(rand_sel)

    best_selection = []
    best_profit = -1.0

    # 2. Refine candidates with local search and record the best
    for sel in candidates:
        if time.time() - start_time > time_limit_s * 0.8:
            break
            
        refined = tools['apply_local_swap_in_out'](sel, time_limit_s=max(0.1, (time_limit_s - (time.time() - start_time)) / len(candidates)))
        
        profit = tools['profit_of_selection'](refined)
        if profit > best_profit:
            best_profit = profit
            best_selection = refined

    # 3. Final ILP attempt if budget allows
    if time_limit_s - (time.time() - start_time) > 1.0:
        ilp_res = tools['ilp_solve_mkp'](time_limit_s=time_limit_s - (time.time() - start_time) - 0.2)
        if ilp_res is not None:
            ilp_profit = tools['profit_of_selection'](ilp_res)
            if ilp_profit > best_profit:
                best_selection = ilp_res

    # 4. Convert selection list to binary vector
    x = [0] * n
    for idx in best_selection:
        if 0 <= idx < n:
            x[idx] = 1
            
    return {'x': x}