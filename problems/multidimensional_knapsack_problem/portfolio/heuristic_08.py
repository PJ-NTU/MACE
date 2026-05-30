# MACE evolved heuristic 08/10 for problem: multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined MKP Solver:
    1. Aggressive Greedy Initialization: Combine multiple greedy strategies.
    2. Local Search: Hill climbing via local swap/add/drop.
    3. ILP-based LNS: Strategic focus on high-impact variables (those with 
       small resource consumption relative to profit density).
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # 1. Initialization: Collect diverse starting points
    candidates = []
    
    # Greedy variants
    candidates.append(tools['greedy_by_profit_density']())
    candidates.append(tools['greedy_by_efficiency']())
    
    # Best of greedy
    best_selection = max(candidates, key=lambda s: tools['profit_of_selection'](s))
    
    # Refine starting point
    best_selection = tools['apply_local_swap_in_out'](best_selection, time_limit_s=max(0.1, time_limit_s * 0.1))
    
    # Calculate profit density for variable selection
    # Items with high profit/resource ratio are 'core' items, others are 'marginal'
    p = instance['p']
    r = instance['r']
    m = instance['m']
    
    def get_density(idx):
        # Average resource consumption across constraints
        avg_res = sum(r[i][idx] / (tools['capacity'](i) + 1e-9) for i in range(m)) / m
        return p[idx] / (avg_res + 1e-9)
    
    densities = [get_density(i) for i in range(n)]
    # Sort items by density to identify "volatile" (low density) vs "core" (high density)
    indexed_densities = sorted(range(n), key=lambda i: densities[i])
    
    # 2. Iterative LNS
    # Focus on re-optimizing the marginal items while keeping core items fixed
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.85:
        iteration += 1
        
        # Adaptive window: focus on the least dense items (the most likely to be swapped)
        window_size = min(n, 20 + (iteration * 10))
        to_optimize_set = set(indexed_densities[:window_size])
        
        # Fix core items (high density) present in best_selection
        # or exclude marginal items not in best_selection
        must_include = [i for i in best_selection if i not in to_optimize_set]
        must_exclude = [i for i in range(n) if i not in to_optimize_set and i not in best_selection]
        
        # Sub-problem solving
        remaining_time = time_limit_s - (time.time() - start_time)
        if remaining_time < 0.2:
            break
            
        sub_limit = min(2.0, remaining_time * 0.4)
        sub_res = tools['ilp_solve_mkp'](
            must_include=must_include, 
            must_exclude=must_exclude, 
            time_limit_s=sub_limit
        )
        
        if sub_res is not None:
            if tools['profit_of_selection'](sub_res) > tools['profit_of_selection'](best_selection):
                best_selection = sub_res
                # Local search refinement on the improved solution
                best_selection = tools['apply_local_swap_in_out'](best_selection, time_limit_s=0.2)
        else:
            # If ILP fails, try a random shuffle/perturbation
            break
            
    # Final output
    x = [0] * n
    for idx in best_selection:
        if 0 <= idx < n:
            x[idx] = 1
            
    return {'x': x}