# MACE evolved heuristic 08/10 for problem: multi_demand_multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized hybrid MDMKP solver:
    Modified to use a weighted mutation strategy: instead of flipping random bits,
    it prioritizes flipping items with low profit-to-resource-consumption ratios
    to escape local optima more intelligently.
    """
    start_time = time.time()
    n = instance['n']
    cost_vec = instance['cost_vector']
    
    # Precompute a proxy for "item quality" to guide mutation
    # Higher quality items are less likely to be flipped to 0, 
    # lower quality items are less likely to be flipped to 1.
    qualities = []
    for i in range(n):
        # Simple heuristic: profit / (avg resource consumption)
        res_sum = sum(instance['A_leq'][k][i] for k in range(instance['m'])) + 1
        qualities.append(cost_vec[i] / res_sum)
    
    best_x = None
    if n <= 100:
        best_x = tools.get('ilp_solve_mdmkp', lambda t: None)(time_limit_s * 0.4)
    
    if best_x is None:
        best_x = tools['greedy_for_demand_then_profit']()
        if not tools['is_feasible']({'x': best_x})[0]:
            best_x = tools['repair_for_demands'](best_x)

    def get_raw_score(x):
        return sum(cost_vec[i] * x[i] for i in range(n))

    best_score = get_raw_score(best_x)

    def refine(x, limit):
        improved = tools['apply_swap_in_out'](x, t_limit=limit)
        if tools['is_feasible']({'x': improved})[0]:
            return improved
        return x

    remaining = time_limit_s - (time.time() - start_time)
    if remaining > 0.1:
        best_x = refine(best_x, remaining * 0.3)
        best_score = get_raw_score(best_x)

    # Weighted mutation: prioritize flipping low-quality items
    weights = [1.0 / (abs(q) + 0.1) for q in qualities]
    
    while time.time() - start_time < time_limit_s - 0.1:
        trial_x = list(best_x)
        # Flip bits weighted by inverse quality
        for _ in range(max(1, n // 20)):
            idx = random.choices(range(n), weights=weights)[0]
            trial_x[idx] = 1 - trial_x[idx]
        
        trial_x = tools['repair_for_demands'](trial_x)
        
        if tools['is_feasible']({'x': trial_x})[0]:
            trial_x = refine(trial_x, 0.05)
            s = get_raw_score(trial_x)
            if s > best_score:
                best_score = s
                best_x = trial_x
        
    return {
        'optimal_value': best_score,
        'x': best_x
    }