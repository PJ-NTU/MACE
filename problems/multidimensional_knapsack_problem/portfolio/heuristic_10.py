# MACE evolved heuristic 10/10 for problem: multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved solver for MKP using a hybrid approach:
    1. Warm start with both greedy strategies.
    2. Local search refinement.
    3. Iterative LNS using ILP sub-problem optimization with an
       efficiency-weighted variable selection strategy.
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # Precompute efficiency for weighted sampling
    p = instance['p']
    r = instance['r']
    m = instance['m']
    efficiencies = [p[j] / sum(r[i][j] for i in range(m)) if sum(r[i][j] for i in range(m)) > 0 else p[j] for j in range(n)]
    
    # 1. Initialization: Greedy baselines
    greedy_density = tools['greedy_by_profit_density']()
    greedy_efficiency = tools['greedy_by_efficiency']()
    
    best_selection = greedy_density if tools['profit_of_selection'](greedy_density) > \
                                       tools['profit_of_selection'](greedy_efficiency) else greedy_efficiency
    
    # 2. Local Search Improvement
    best_selection = tools['apply_local_swap_in_out'](best_selection, time_limit_s=max(0.1, time_limit_s * 0.1))
    
    # 3. ILP-based Large Neighborhood Search (LNS)
    if time.time() - start_time < time_limit_s * 0.3:
        ilp_full = tools['ilp_solve_mkp'](time_limit_s=max(0.5, time_limit_s * 0.2))
        if ilp_full is not None:
            if tools['profit_of_selection'](ilp_full) > tools['profit_of_selection'](best_selection):
                best_selection = ilp_full
    
    # Iterative LNS: Weighted Mutation
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.9:
        iteration += 1
        window_size = min(n, 40 + (iteration * 5))
        
        # Weighted sampling: items with lower efficiency are more likely to be in the 'to_optimize' set
        # Using random.choices for weighted sampling instead of random.sample with counts
        weights = [1.0 / (e + 1e-9) for e in efficiencies]
        to_optimize = random.choices(range(n), weights=weights, k=min(n, window_size))
        
        to_optimize_set = set(to_optimize)
        must_include = [i for i in best_selection if i not in to_optimize_set]
        must_exclude = [i for i in range(n) if i not in to_optimize_set and i not in best_selection]
        
        sub_limit = min(1.5, (time_limit_s - (time.time() - start_time)) * 0.5)
        sub_res = tools['ilp_solve_mkp'](
            must_include=must_include, 
            must_exclude=must_exclude, 
            time_limit_s=sub_limit
        )
        
        if sub_res is not None:
            if tools['profit_of_selection'](sub_res) > tools['profit_of_selection'](best_selection):
                best_selection = sub_res
        
        if time.time() - start_time > time_limit_s * 0.95:
            break
            
    x = [0] * n
    for idx in best_selection:
        if 0 <= idx < n:
            x[idx] = 1
            
    return {'x': x}