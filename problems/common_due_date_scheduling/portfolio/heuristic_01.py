# MACE evolved heuristic 01/10 for problem: common_due_date_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Common Due Date Scheduling problem using a V-shape initialization
    followed by a combination of local search heuristics (Or-opt and 2-opt).
    """
    start_time = time.time()
    
    # 1. Warm start: Use the V-shape construction heuristic
    # This is known to be theoretically strong for this specific problem class.
    best_schedule = tools['v_shape_construct']()
    best_penalty = tools['compute_total_penalty'](best_schedule)
    
    # 2. Local Search refinement
    # We alternate between insertion (Or-opt) and 2-opt swaps to explore the
    # neighborhood space efficiently within the remaining time budget.
    
    time_budget_for_search = time_limit_s * 0.95
    
    while time.time() - start_time < time_budget_for_search:
        # Save current best to check for improvements
        current_best = best_penalty
        
        # Apply insertion-based search (Or-opt)
        # This is particularly good at moving jobs across the due date boundary.
        improved_schedule = tools['apply_insertion_search'](
            best_schedule, 
            time_limit_s=max(0.1, (time_budget_for_search - (time.time() - start_time)) / 2)
        )
        
        # Apply swap-based search (2-opt)
        # This is good for fine-tuning the sequence.
        final_schedule = tools['apply_swap_2opt'](
            improved_schedule, 
            time_limit_s=max(0.1, (time_budget_for_search - (time.time() - start_time)))
        )
        
        new_penalty = tools['compute_total_penalty'](final_schedule)
        
        if new_penalty < best_penalty:
            best_penalty = new_penalty
            best_schedule = final_schedule
        
        # If no improvement was found in this pass, break to avoid infinite loops
        if best_penalty >= current_best:
            break
            
    return {'schedule': list(best_schedule)}