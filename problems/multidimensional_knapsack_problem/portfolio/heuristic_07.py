# MACE evolved heuristic 07/10 for problem: multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved solver for MKP using a hybrid approach:
    1. Warm start with both greedy strategies.
    2. Local search refinement.
    3. Iterative LNS using ILP sub-problem optimization.
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # 1. Initialization: Greedy baselines
    greedy_density = tools['greedy_by_profit_density']()
    greedy_efficiency = tools['greedy_by_efficiency']()
    
    best_selection = greedy_density if tools['profit_of_selection'](greedy_density) > \
                                       tools['profit_of_selection'](greedy_efficiency) else greedy_efficiency
    
    # 2. Local Search Improvement
    # Crucial for reaching local optima quickly before expensive ILP calls.
    best_selection = tools['apply_local_swap_in_out'](best_selection, time_limit_s=max(0.1, time_limit_s * 0.1))
    
    # 3. ILP-based Large Neighborhood Search (LNS)
    # Focuses on improving the current best by solving restricted sub-problems.
    # We prioritize ILP for the full solve initially if time permits.
    if time.time() - start_time < time_limit_s * 0.3:
        ilp_full = tools['ilp_solve_mkp'](time_limit_s=max(0.5, time_limit_s * 0.2))
        if ilp_full is not None:
            if tools['profit_of_selection'](ilp_full) > tools['profit_of_selection'](best_selection):
                best_selection = ilp_full
    
    # Iterative LNS: Fix a subset, optimize the rest.
    # We use a sliding window/random subset strategy to navigate the search space.
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.9:
        iteration += 1
        # Adaptive window size: start small, increase if time allows.
        window_size = min(n, 30 + (iteration * 5))
        
        # Decide which variables to lock (keep 90% fixed)
        all_items = list(range(n))
        to_optimize = random.sample(all_items, k=min(n, window_size))
        to_optimize_set = set(to_optimize)
        
        must_include = [i for i in best_selection if i not in to_optimize_set]
        must_exclude = [i for i in all_items if i not in to_optimize_set and i not in best_selection]
        
        # Solve sub-problem
        sub_limit = min(1.5, (time_limit_s - (time.time() - start_time)) * 0.5)
        sub_res = tools['ilp_solve_mkp'](
            must_include=must_include, 
            must_exclude=must_exclude, 
            time_limit_s=sub_limit
        )
        
        if sub_res is not None:
            if tools['profit_of_selection'](sub_res) > tools['profit_of_selection'](best_selection):
                best_selection = sub_res
        
        # Backoff if we are running out of time
        if time.time() - start_time > time_limit_s * 0.95:
            break
            
    # Final conversion to binary decision list
    x = [0] * n
    for idx in best_selection:
        if 0 <= idx < n:
            x[idx] = 1
            
    return {'x': x}