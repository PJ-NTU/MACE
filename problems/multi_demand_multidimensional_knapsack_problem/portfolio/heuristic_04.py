# MACE evolved heuristic 04/10 for problem: multi_demand_multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized hybrid MDMKP solver:
    1. Uses ILP for small/tight instances where it is most effective.
    2. Uses greedy construction followed by a robust local search.
    3. Implements a restart mechanism with bit-flip diversification to escape
       local optima, maintaining feasibility constraints throughout.
    """
    start_time = time.time()
    n = instance['n']
    
    # Heuristic: ILP is excellent for small n or very tight constraints.
    # The time limit for ILP is set to be aggressive to leave room for local search.
    best_x = None
    if n <= 100:
        best_x = tools.get('ilp_solve_mdmkp', lambda t: None)(time_limit_s * 0.4)
    
    # If ILP didn't return a result, use high-quality greedy construction.
    if best_x is None:
        best_x = tools['greedy_for_demand_then_profit']()
        # Verify and repair if necessary
        if not tools['is_feasible']({'x': best_x})[0]:
            best_x = tools['repair_for_demands'](best_x)

    def get_raw_score(x):
        return sum(instance['cost_vector'][i] * x[i] for i in range(n))

    best_score = get_raw_score(best_x)

    # Local search refinement
    def refine(x, limit):
        improved = tools['apply_swap_in_out'](x, t_limit=limit)
        if tools['is_feasible']({'x': improved})[0]:
            return improved
        return x

    # Initial refinement
    remaining = time_limit_s - (time.time() - start_time)
    if remaining > 0.1:
        best_x = refine(best_x, remaining * 0.3)
        best_score = get_raw_score(best_x)

    # Hill-climbing with random restarts
    # Diversify by flipping bits and repairing to explore the feasible region
    while time.time() - start_time < time_limit_s - 0.1:
        trial_x = list(best_x)
        # Flip a few random bits to jump to a different region
        for _ in range(max(1, n // 20)):
            idx = random.randint(0, n - 1)
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