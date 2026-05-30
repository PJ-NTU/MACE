# MACE evolved heuristic 03/10 for problem: equitable_partitioning_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the equitable partitioning problem using a GRASP-inspired 
    constructive heuristic followed by a local search refinement.
    """
    start_time = time.time()
    
    # 1. Initialization
    n = tools['n_items']()
    k = tools['k_groups']()
    
    # Use greedy_balanced_split for a strong initial construction
    # This is usually very effective for balancing attribute distributions.
    # We use a small time buffer to ensure we have time for local search.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time < 0.1:
        remaining_time = 0.1
        
    # Attempt to construct a good initial solution
    try:
        assignment = tools['greedy_balanced_split'](seed=42)
    except Exception:
        # Fallback to round-robin if greedy fails
        assignment = tools['round_robin_partition'](seed=42)
    
    # 2. Local Search Refinement
    # apply_swap_items_across_groups is a robust local search that adheres 
    # to the 8-group constraint and improves the total imbalance.
    # We allocate the majority of the remaining time to this search.
    
    # Ensure the partition is valid for the tool
    if not tools['is_valid_partition'](assignment):
        assignment = tools['round_robin_partition'](seed=random.randint(0, 1000))
        
    try:
        # Refine the solution using the provided tool, respecting the time limit
        refined_assignment = tools['apply_swap_items_across_groups'](
            partition=assignment, 
            time_limit_s=max(0.1, time_limit_s - (time.time() - start_time) - 0.5),
            seed=random.randint(0, 1000)
        )
        assignment = refined_assignment
    except Exception:
        # If refinement fails, proceed with the construction-based assignment
        pass
    
    # Final check before returning
    feasible, _ = tools['is_feasible']({"assignment": assignment})
    if not feasible:
        # Emergency fallback
        assignment = tools['round_robin_partition'](seed=123)
        
    return {"assignment": assignment}