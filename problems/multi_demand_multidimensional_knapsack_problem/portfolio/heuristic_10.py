# MACE evolved heuristic 10/10 for problem: multi_demand_multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Redesigned MDMKP solver:
    Uses a multi-start greedy approach followed by aggressive local search.
    ILP is used as a final refinement step if time permits.
    """
    start_time = time.time()
    n = instance['n']
    
    best_x = None
    best_score = -float('inf')

    def get_score(x):
        return sum(instance['cost_vector'][i] * x[i] for i in range(n))

    # Phase 1: Multi-start construction
    # Use greedy_for_demand_then_profit as a base, add randomized variants
    start_iters = 0
    while time.time() - start_time < time_limit_s * 0.3:
        if start_iters == 0:
            x = tools['greedy_for_demand_then_profit']()
        else:
            # Randomized greedy: shuffle item priorities
            x = [0] * n
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices:
                x[i] = 1
                feasible, _ = tools['is_feasible']({"x": x})
                if not feasible:
                    x[i] = 0
            x = tools['repair_for_demands'](x)
        
        # Local search refinement
        remaining = time_limit_s - (time.time() - start_time)
        if remaining > 0.1:
            x = tools['apply_swap_in_out'](x, t_limit=min(remaining, 1.0))
        
        feasible, _ = tools['is_feasible']({"x": x})
        if feasible:
            score = get_score(x)
            if score > best_score:
                best_score = score
                best_x = x
        start_iters += 1

    # Phase 2: ILP refinement
    # If we have significant time left, try to improve via exact solver
    if time.time() - start_time < time_limit_s * 0.9:
        ilp_time = time_limit_s - (time.time() - start_time) - 0.2
        if ilp_time > 1.0:
            x_ilp = tools.get('ilp_solve_mdmkp', lambda t: None)(time_limit_s=ilp_time)
            if x_ilp is not None:
                feasible, _ = tools['is_feasible']({"x": x_ilp})
                if feasible:
                    score = get_score(x_ilp)
                    if score > best_score:
                        best_score = score
                        best_x = x_ilp

    # Final fallback
    if best_x is None:
        best_x = [0] * n
        best_score = 0

    return {
        "optimal_value": best_score,
        "x": best_x
    }