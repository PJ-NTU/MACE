# MACE evolved heuristic 01/10 for problem: multi_demand_multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    MDMKP solver using a hybrid approach:
    1. ILP solver (CBC) as the primary engine (often optimal for these constraints).
    2. Fallback to greedy construction + local search refinement if ILP fails.
    """
    start_time = time.time()
    n = instance['n']
    
    # 1. Try ILP solver first (it handles both <= and >= constraints natively).
    # We allocate 80% of the time limit to the ILP solver.
    ilp_limit = time_limit_s * 0.8
    try:
        x_ilp = tools['ilp_solve_mdmkp'](time_limit_s=ilp_limit)
        if x_ilp is not None:
            is_feas, _ = tools['is_feasible']({'x': x_ilp})
            if is_feas:
                return {'x': x_ilp, 'optimal_value': tools['objective']({'x': x_ilp})}
    except Exception:
        pass

    # 2. Heuristic construction: greedy_for_demand_then_profit
    # This ensures we satisfy the >= constraints first.
    x_best = tools['greedy_for_demand_then_profit']()
    
    # Check feasibility of constructed solution
    is_feas, _ = tools['is_feasible']({'x': x_best})
    if not is_feas:
        # If construction failed, try to repair
        x_best = tools['repair_for_demands'](x_best)
        is_feas, _ = tools['is_feasible']({'x': x_best})
        if not is_feas:
            # Fallback to zero vector if nothing else works
            x_best = [0] * n

    # 3. Refine with local search
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        x_refined = tools['apply_swap_in_out'](x_best, t_limit=remaining_time)
        # Verify refinement kept it feasible
        if tools['is_feasible']({'x': x_refined})[0]:
            x_best = x_refined

    # Final result
    score = tools['objective']({'x': x_best})
    return {'x': x_best, 'optimal_value': score}