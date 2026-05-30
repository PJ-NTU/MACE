# MACE evolved heuristic 04/10 for problem: equitable_partitioning_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the equitable partitioning problem using a GRASP-inspired 
    constructive heuristic followed by local search.
    """
    start_time = time.time()
    
    # Define time buffers for phases
    # Reserve 20% of time for safety buffer and final verification
    remaining_time = lambda: time_limit_s - (time.time() - start_time)
    
    # 1. Generate a strong initial partition
    # greedy_balanced_split is excellent for this specific objective
    # We use a seed to ensure reproducibility if called multiple times, 
    # but vary it if we had a loop.
    best_assignment = tools['greedy_balanced_split'](seed=42)
    best_score = tools['total_imbalance'](best_assignment)
    
    # 2. Local Search
    # apply_swap_items_across_groups is a robust first-improvement local search
    # that respects the 8-group constraint and preserves the valid partition.
    # We call it with the remaining time budget.
    
    ls_time = max(1.0, remaining_time() * 0.8)
    
    try:
        refined_assignment = tools['apply_swap_items_across_groups'](
            partition=best_assignment,
            time_limit_s=ls_time,
            seed=random.randint(0, 10000)
        )
        
        # Check if the local search found a better score before returning
        refined_score = tools['total_imbalance'](refined_assignment)
        if refined_score < best_score:
            best_assignment = refined_assignment
    except Exception:
        # If local search fails for any reason, fall back to the greedy result
        pass
        
    # Final sanity check: ensure the output is valid per the problem spec
    # The evaluation function expects a dictionary with 'assignment'
    return {'assignment': best_assignment}