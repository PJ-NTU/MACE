# MACE evolved heuristic 06/10 for problem: multi_demand_multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for MDMKP.
    
    Hypothesis:
    - High-density constraints (m+q)/n > 0.3 act as strong filters that ILP solvers 
      can prune effectively using branch-and-bound.
    - Low-density, large-scale instances have a flatter, more complex search space 
      where constructive heuristics paired with local search are more robust against 
      the "stalling" behavior of exact solvers.
    """
    start_time = time.time()
    n = instance['n']
    m = instance['m']
    q = instance['q']
    
    # Structural feature: constraint density
    # Instances with many constraints relative to n are usually "tightly 
    # constrained", where ILP excels at finding the exact boundary.
    density = (m + q) / n if n > 0 else 0
    
    # Strategy 1: ILP-Heavy (for tightly constrained or small problems)
    # If the problem is small enough for exact methods or dense enough that
    # search space pruning is efficient, we commit to the exact solver.
    if n <= 150 or density > 0.3:
        ilp_res = tools['ilp_solve_mdmkp'](time_limit_s=time_limit_s * 0.9)
        if ilp_res is not None:
            sol = {"x": ilp_res}
            if tools['is_feasible'](sol)[0]:
                return {"optimal_value": tools['objective'](sol), "x": ilp_res}

    # Strategy 2: Heuristic Construction + Local Search (for sparse/large problems)
    # We use a multi-start strategy to navigate the search space.
    best_x = [0] * n
    best_obj_val = -float('inf')
    
    # Attempt multiple construction passes
    while time.time() - start_time < time_limit_s * 0.8:
        # Constructive phase: prioritize demand satisfaction
        candidate = tools['greedy_for_demand_then_profit']()
        
        # Ensure feasibility
        if not tools['is_feasible']({'x': candidate})[0]:
            candidate = tools['repair_for_demands'](candidate)
            
        # Refinement phase: local search with time budget
        if tools['is_feasible']({'x': candidate})[0]:
            remaining = time_limit_s - (time.time() - start_time)
            if remaining > 0.1:
                candidate = tools['apply_swap_in_out'](candidate, t_limit=min(remaining, 0.5))
            
            # Check objective
            current_obj = sum(instance['cost_vector'][i] * candidate[i] for i in range(n))
            if current_obj > best_obj_val:
                best_obj_val = current_obj
                best_x = candidate
        
        # If we found a decent solution and are nearing the limit, stop
        if best_obj_val > 0 and time.time() - start_time > time_limit_s * 0.6:
            break

    # Fallback to a single greedy construction if all else fails
    if best_obj_val == -float('inf'):
        best_x = tools['repair_for_demands'](tools['greedy_for_demand_then_profit']())
        best_obj_val = sum(instance['cost_vector'][i] * best_x[i] for i in range(n))

    return {
        "optimal_value": best_obj_val,
        "x": best_x
    }