# MACE evolved heuristic 01/10 for problem: multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the multidimensional knapsack problem using a hybrid approach:
    1. Warm start with two greedy strategies.
    2. Local search (swap/add) to reach local optima.
    3. If time permits, iterative random perturbations followed by repair
       to explore the search space.
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # Generate initial candidates
    best_selection = []
    best_profit = -1.0
    
    candidates = [
        tools['greedy_by_profit_density'](),
        tools['greedy_by_efficiency']()
    ]
    
    # Improve candidates with local search
    for seed in candidates:
        if time.time() - start_time > time_limit_s * 0.2:
            break
        
        improved = tools['apply_local_swap_in_out'](seed, time_limit_s=(time_limit_s * 0.1))
        profit = tools['profit_of_selection'](improved)
        if profit > best_profit:
            best_profit = profit
            best_selection = improved

    # If we have remaining time, try Large Neighborhood Search (LNS)
    # Destroy: randomly remove items, Repair: greedy repair
    while time.time() - start_time < time_limit_s * 0.9:
        if not best_selection:
            break
            
        # Destroy
        num_to_remove = random.randint(1, max(1, len(best_selection) // 4))
        indices_to_remove = random.sample(range(len(best_selection)), num_to_remove)
        current = [best_selection[i] for i in range(len(best_selection)) if i not in indices_to_remove]
        
        # Repair
        repaired = tools['repair_capacity_violation'](current)
        
        # Local Search Improvement
        improved = tools['apply_local_swap_in_out'](repaired, time_limit_s=(time_limit_s * 0.05))
        
        profit = tools['profit_of_selection'](improved)
        if profit > best_profit:
            best_profit = profit
            best_selection = improved
            
    # Convert selection to binary list
    x = [0] * n
    for idx in best_selection:
        if 0 <= idx < n:
            x[idx] = 1
            
    return {'x': x}