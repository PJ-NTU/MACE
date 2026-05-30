# MACE evolved heuristic 07/10 for problem: multi_demand_multidimensional_knapsack_problem
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined MDMKP solver:
    Uses ILP for global structure if time permits, followed by a robust
    local search (swap/flip) that respects both capacity and demand constraints.
    """
    start_time = time.time()
    n = instance['n']
    cost_vector = instance['cost_vector']

    def get_score(x):
        return sum(cost_vector[i] * x[i] for i in range(n))

    # 1. Initial Construction
    # Start with the strongest base heuristic provided by the framework
    x_best = tools['greedy_for_demand_then_profit']()
    
    # Ensure baseline feasibility
    if not tools['is_feasible']({'x': x_best})[0]:
        x_best = tools['repair_for_demands'](x_best)
        # If repair fails to meet capacity, force capacity feasibility
        if not tools['is_caps_satisfied'](x_best):
            for i in range(n):
                if x_best[i] == 1 and not tools['is_caps_satisfied'](x_best):
                    x_best[i] = 0
    
    current_best_score = get_score(x_best)

    # 2. ILP Warm-start
    # If time allows, attempt an exact solve to find a better baseline
    if time_limit_s > 2.0:
        ilp_budget = min(time_limit_s * 0.3, 5.0)
        try:
            x_ilp = tools['ilp_solve_mdmkp'](time_limit_s=ilp_budget)
            if x_ilp is not None:
                score_ilp = get_score(x_ilp)
                if score_ilp > current_best_score:
                    x_best = x_ilp
                    current_best_score = score_ilp
        except Exception:
            pass

    # 3. Local Search Refinement
    # Use the provide swap tool as the primary engine for local improvement
    # It is highly optimized and respects the MDMKP constraint structure.
    while time.time() - start_time < time_limit_s - 0.5:
        # Small perturbation: flip 1-2 bits to escape local optima
        x_trial = list(x_best)
        for _ in range(random.randint(1, 2)):
            idx = random.randint(0, n - 1)
            x_trial[idx] = 1 - x_trial[idx]
        
        # Repair and refine
        x_trial = tools['repair_for_demands'](x_trial)
        
        # Only proceed to expensive swap if capacity is met
        if tools['is_caps_satisfied'](x_trial):
            refine_budget = min(0.5, (time_limit_s - (time.time() - start_time)) * 0.5)
            x_refined = tools['apply_swap_in_out'](x_trial, t_limit=refine_budget)
            
            score = get_score(x_refined)
            if score > current_best_score:
                current_best_score = score
                x_best = x_refined
        
        # Quick exit if minimal time left
        if time.time() - start_time > time_limit_s - 0.1:
            break

    return {
        "optimal_value": current_best_score,
        "x": x_best
    }