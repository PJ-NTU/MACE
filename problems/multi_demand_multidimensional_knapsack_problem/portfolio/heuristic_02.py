# MACE evolved heuristic 02/10 for problem: multi_demand_multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    MDMKP solver using a hybrid approach:
    1. ILP solver (CBC) as the primary engine for optimal/near-optimal results.
    2. Fallback to greedy construction + local search if ILP is unavailable 
       or times out.
    """
    start_time = time.time()
    n = instance['n']
    
    # 1. Attempt exact solution via ILP if available
    try:
        # Give ILP 80% of the available time budget
        ilp_time = max(1.0, time_limit_s * 0.8)
        x_ilp = tools.get('ilp_solve_mdmkp', lambda t: None)(time_limit_s=ilp_time)
        
        if x_ilp is not None:
            # Verify feasibility
            sol = {"x": x_ilp}
            feasible, _ = tools['is_feasible'](sol)
            if feasible:
                return {"optimal_value": tools['objective'](sol), "x": x_ilp}
    except Exception:
        pass

    # 2. Heuristic fallback: Greedy Construction + Local Search
    # Start with Greedy for demand satisfaction
    x = tools['greedy_for_demand_then_profit']()
    
    # Ensure it's feasible (repair if necessary)
    if not tools['is_caps_satisfied'](x) or not tools['is_demands_satisfied'](x):
        # Fallback to repair
        x = tools['repair_for_demands'](x)
        # If still not cap-feasible, try a simpler construction
        if not tools['is_caps_satisfied'](x):
            x = [0] * n
            # Simple greedy: add items with positive profit that don't violate caps
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices:
                x_test = x[:]
                x_test[i] = 1
                if tools['is_caps_satisfied'](x_test):
                    x = x_test
            x = tools['repair_for_demands'](x)

    # 3. Refine with Local Search
    elapsed = time.time() - start_time
    remaining_time = max(0.1, time_limit_s - elapsed - 0.5)
    
    x_refined = tools['apply_swap_in_out'](x, t_limit=remaining_time)
    
    # Final check
    final_sol = {"x": x_refined}
    feasible, _ = tools['is_feasible'](final_sol)
    
    if not feasible:
        # If refined solution is invalid, return the original greedy/repaired one
        final_sol = {"x": x}
        
    return {
        "optimal_value": tools['objective'](final_sol),
        "x": final_sol["x"]
    }