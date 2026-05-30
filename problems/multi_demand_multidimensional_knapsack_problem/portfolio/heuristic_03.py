# MACE evolved heuristic 03/10 for problem: multi_demand_multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    MDMKP solver using a robust multi-start strategy:
    1. Prioritizes the exact ILP solver for optimality within a strict time budget.
    2. Employs a multi-start heuristic construction phase using randomized greedy 
       approaches to generate diverse feasible candidates.
    3. Uses local search refinement to optimize the best candidate found.
    """
    start_time = time.time()
    n = instance['n']
    
    # 1. Exact Solver
    ilp_result = None
    if 'ilp_solve_mdmkp' in tools:
        ilp_result = tools['ilp_solve_mdmkp'](time_limit_s * 0.6)
    
    if ilp_result is not None:
        best_x = ilp_result
    else:
        # 2. Multi-start Heuristic Construction
        best_x = [0] * n
        best_score = -float('inf')
        
        # Ensure we have at least one feasible starting point
        initial_x = tools['greedy_for_demand_then_profit']()
        if tools['is_feasible']({"x": initial_x})[0]:
            best_x = initial_x
            best_score = tools['objective']({"x": best_x})
            # objective() returns 1/raw_score, so we need to invert for comparison
            # or simply use the raw score if available. Since we need raw score:
            best_score = 1.0 / best_score
        
        while time.time() - start_time < time_limit_s * 0.75:
            current_x = tools['greedy_for_demand_then_profit']()
            
            # Ensure feasibility: Repair if standard construction fails
            if not tools['is_feasible']({"x": current_x})[0]:
                current_x = tools['repair_for_demands'](current_x)
            
            # Final check before updating best_x
            if tools['is_feasible']({"x": current_x})[0]:
                current_score = 1.0 / tools['objective']({"x": current_x})
                if current_score > best_score:
                    best_score = current_score
                    best_x = current_x
            
            if best_score > 0 and random.random() < 0.3:
                break

    # 3. Local Search Refinement
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1 and tools['is_feasible']({"x": best_x})[0]:
        refined_x = tools['apply_swap_in_out'](best_x, t_limit=remaining_time * 0.9)
        if tools['is_feasible']({"x": refined_x})[0]:
            best_x = refined_x

    # Final validation ensure we return a feasible solution
    feasible, _ = tools['is_feasible']({"x": best_x})
    if not feasible:
        best_x = [0] * n
        if tools['is_feasible']({"x": best_x})[0]:
            score = 1.0 / tools['objective']({"x": best_x})
        else:
            score = 0
    else:
        score = 1.0 / tools['objective']({"x": best_x})

    return {
        "x": best_x,
        "optimal_value": score
    }