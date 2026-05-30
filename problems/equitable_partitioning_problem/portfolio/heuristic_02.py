# MACE evolved heuristic 02/10 for problem: equitable_partitioning_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the equitable partitioning problem using a greedy construction 
    followed by a local search refinement.
    """
    start_time = time.time()
    
    # 1. Initialization
    # Use greedy_balanced_split as a high-quality warm start.
    # It ensures the 8-group constraint and balances group sizes.
    try:
        current_assignment = tools['greedy_balanced_split'](seed=42)
    except Exception:
        # Fallback to round-robin if greedy fails
        current_assignment = tools['round_robin_partition'](seed=42)
        
    # 2. Local Search
    # Refine the solution using apply_swap_items_across_groups.
    # This tool is designed to optimize the exact imbalance metric 
    # while preserving feasibility.
    
    # We leave a buffer of 0.5s for overhead and final processing.
    remaining_time = time_limit_s - (time.time() - start_time) - 0.5
    
    if remaining_time > 0.1:
        refined_assignment = tools['apply_swap_items_across_groups'](
            partition=current_assignment,
            time_limit_s=remaining_time,
            seed=random.randint(0, 10000)
        )
    else:
        refined_assignment = current_assignment

    # 3. Final Verification
    # Ensure the output format is correct and feasible.
    solution = {'assignment': refined_assignment}
    
    is_ok, msg = tools['is_feasible'](solution)
    if not is_ok:
        # If something went wrong, fall back to the safest construction
        return {'assignment': tools['round_robin_partition'](seed=123)}
        
    return solution