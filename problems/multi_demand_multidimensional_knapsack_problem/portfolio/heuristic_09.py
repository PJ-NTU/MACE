# MACE evolved heuristic 09/10 for problem: multi_demand_multidimensional_knapsack_problem
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style hybrid MDMKP solver with enforced feasibility checks.
    """
    start_time = time.time()
    n = instance['n']
    m = instance['m']
    q = instance['q']
    cost_vector = instance['cost_vector']

    def calculate_score(x):
        return sum(cost_vector[i] * x[i] for i in range(n))

    # Feature: Constraint Density
    density = (m + q) / n
    is_sparse = density < 0.35 

    x_best = None
    best_val = -float('inf')

    # REGIME 1: Sparse / Large Instances (B-style)
    if is_sparse and time_limit_s > 1.0:
        ilp_budget = min(time_limit_s * 0.5, 6.0)
        try:
            x_ilp = tools['ilp_solve_mdmkp'](time_limit_s=ilp_budget)
            if x_ilp is not None and tools['is_feasible']({'x': x_ilp})[0]:
                x_best = x_ilp
                best_val = calculate_score(x_best)
        except Exception:
            pass
    
    # Construction phase
    if x_best is None:
        x_best = tools['greedy_for_demand_then_profit']()
        # Ensure feasibility: repair if necessary
        if not tools['is_feasible']({'x': x_best})[0]:
            x_best = tools['repair_for_demands'](x_best)
            # If still not feasible (capacity), prune greedily
            if not tools['is_caps_satisfied'](x_best):
                for i in range(n):
                    x_best[i] = 0
                    if tools['is_caps_satisfied'](x_best):
                        break
                x_best = tools['repair_for_demands'](x_best)
        
        # Final safety check
        if not tools['is_feasible']({'x': x_best})[0]:
            # Emergency fallback: zero vector is usually infeasible for >=,
            # but we return it to avoid crashing if absolutely no solution found
            x_best = [0] * n
            best_val = -float('inf')
        else:
            best_val = calculate_score(x_best)

    # REGIME 2: Iterative Refinement
    while time.time() - start_time < time_limit_s - 0.2:
        x_trial = list(x_best)
        
        num_flips = max(1, int(n * 0.05)) if (time.time() - start_time) < (time_limit_s * 0.5) else 1
        for _ in range(num_flips):
            idx = random.randint(0, n - 1)
            x_trial[idx] = 1 - x_trial[idx]
        
        # Ensure feasibility of trial
        x_trial = tools['repair_for_demands'](x_trial)
        if not tools['is_caps_satisfied'](x_trial):
            continue
            
        # Local improvement
        refine_time = min(0.25, (time_limit_s - (time.time() - start_time)) * 0.3)
        x_refined = tools['apply_swap_in_out'](x_trial, t_limit=refine_time)
        
        if tools['is_feasible']({'x': x_refined})[0]:
            score = calculate_score(x_refined)
            if score > best_val:
                best_val = score
                x_best = x_refined
                
    return {
        "optimal_value": best_val,
        "x": x_best
    }