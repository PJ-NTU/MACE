# MACE evolved heuristic 09/10 for problem: set_covering
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized Set Covering solver using a multi-stage approach:
    1. Robust initial greedy construction.
    2. Local search (redundancy removal).
    3. ILP-based Large Neighborhood Search (LNS) with adaptive time allocation.
    """
    start_time = time.time()
    
    # 1. Initial High-Quality Construction
    best_solution = tools['greedy_cover_by_cost_ratio']()
    best_solution = tools['remove_redundant'](best_solution)
    best_cost = tools['cost_of_selection'](best_solution)
    
    # 2. LNS Strategy
    # Systemic improvement: Use ILP as a powerful repair/improvement operator.
    # We restrict the search space by forcing a subset of the current best columns,
    # then allow the ILP to optimize the remainder.
    
    # Time allocation: 
    # Reserve 20% for final cleanup, distribute remaining across LNS iterations.
    iterations = 0
    max_iterations = 5
    
    while time.time() - start_time < time_limit_s * 0.80 and iterations < max_iterations:
        current_cols = list(best_solution)
        # Adaptive LNS: remove a larger percentage if we are early, 
        # smaller if we are late to converge.
        drop_rate = 0.3 if iterations < 2 else 0.15
        num_to_remove = max(1, int(len(current_cols) * drop_rate))
        
        random.shuffle(current_cols)
        kept_cols = current_cols[num_to_remove:]
        
        # Calculate remaining budget for this ILP step
        remaining_time = time_limit_s * 0.90 - (time.time() - start_time)
        if remaining_time <= 0.1:
            break
            
        # ILP identifies the optimal completion given the forced columns
        ilp_sol = tools['ilp_solve_cover'](
            must_include=kept_cols, 
            time_limit_s=remaining_time / (max_iterations - iterations)
        )
        
        if ilp_sol:
            refined = tools['remove_redundant'](ilp_sol)
            cost = tools['cost_of_selection'](refined)
            if cost < best_cost:
                best_cost = cost
                best_solution = refined
                
        iterations += 1
        
    # 3. Final Polish
    # Ensure no redundant columns exist in the best found solution.
    best_solution = tools['remove_redundant'](best_solution)
    
    # Final sanity check: if something went wrong, ensure we return a valid cover.
    if not tools['is_full_cover'](best_solution):
        # Fallback to default if LNS somehow corrupted the cover
        return tools['solve_default'](time_limit_s=max(0.1, time_limit_s - (time.time() - start_time)))
        
    return tools['make_solution'](best_solution)