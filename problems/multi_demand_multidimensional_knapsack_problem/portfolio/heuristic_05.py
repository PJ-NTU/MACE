# MACE evolved heuristic 05/10 for problem: multi_demand_multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    MDMKP solver:
    1. Prioritizes ILP (CBC) as it is highly effective for these constraints.
    2. Uses a 'best-of-both-worlds' strategy:
       - If ILP finds a solution, we verify and return it as the strong baseline.
       - If ILP fails or is incomplete, we use a deterministic priority construction
         followed by a focused, time-bounded local search.
    """
    start_time = time.time()
    n = instance['n']
    m = instance['m']
    q = instance['q']
    
    best_x = None
    best_obj = float('inf')

    # 1. ILP Phase: Maximize time allocation to the exact solver (90% budget)
    # The ILP solver is the strongest component for this specific problem class.
    ilp_limit = time_limit_s * 0.9
    try:
        x_ilp = tools['ilp_solve_mdmkp'](time_limit_s=ilp_limit)
        if x_ilp is not None:
            sol = {"x": x_ilp}
            feasible, _ = tools['is_feasible'](sol)
            if feasible:
                best_x = x_ilp
                best_obj = tools['objective'](sol)
    except Exception:
        pass

    # 2. Heuristic Phase: Fallback / Refinement
    # If ILP didn't find a solution, or to attempt an improvement, use
    # a robust, deterministic two-phase construction.
    if best_x is None:
        # Phase A: Initial construction
        candidate = tools['greedy_for_demand_then_profit']()
        
        # Check feasibility and repair if necessary
        feasible, _ = tools['is_feasible']({'x': candidate})
        if not feasible:
            candidate = tools['repair_for_demands'](candidate)
            feasible, _ = tools['is_feasible']({'x': candidate})
        
        if feasible:
            best_x = candidate
            best_obj = tools['objective']({'x': best_x})
        else:
            # Absolute fallback: zero vector
            best_x = [0] * n
            best_obj = tools['objective']({'x': best_x})

    # 3. Local Search Refinement
    # Only perform local search if we still have significant time remaining.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.2:
        refined = tools['apply_swap_in_out'](best_x, t_limit=remaining_time)
        if tools['is_feasible']({'x': refined})[0]:
            refined_obj = tools['objective']({'x': refined})
            if refined_obj < best_obj:
                best_x = refined
                best_obj = refined_obj

    return {
        "optimal_value": best_obj,
        "x": best_x
    }