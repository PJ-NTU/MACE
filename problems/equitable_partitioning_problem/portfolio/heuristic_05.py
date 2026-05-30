# MACE evolved heuristic 05/10 for problem: equitable_partitioning_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the equitable partitioning problem using a GRASP-inspired 
    approach: Greedy construction followed by iterative local search (Hill Climbing).
    """
    start_time = time.time()
    
    # 1. Initialization
    # Use greedy construction as a strong baseline
    # The greedy_balanced_split is robust and respects the 8-group constraint.
    best_assignment = tools['greedy_balanced_split'](seed=42)
    best_score = tools['total_imbalance'](best_assignment)
    
    # 2. Local Search
    # Refine the partition using apply_swap_items_across_groups.
    # This tool is optimized for the objective function and respects feasibility.
    # We allocate 80% of the remaining time to this local search.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        refined_assignment = tools['apply_swap_items_across_groups'](
            partition=best_assignment,
            time_limit_s=remaining_time * 0.8,
            seed=random.randint(0, 10000)
        )
        
        refined_score = tools['total_imbalance'](refined_assignment)
        
        if refined_score < best_score:
            best_assignment = refined_assignment
            best_score = refined_score

    # 3. Final sanity check and return
    # Ensure the result is formatted as expected by the evaluation function
    # The tools guarantee feasibility if the initial partition was valid.
    return {'assignment': best_assignment}