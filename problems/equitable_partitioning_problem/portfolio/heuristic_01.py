# MACE evolved heuristic 01/10 for problem: equitable_partitioning_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve the equitable partitioning problem using a multi-stage approach:
    1. Start with a greedy construction to satisfy the 8-group constraint.
    2. Refine the partition using iterative local search (swap and move).
    3. Monitor execution time to ensure return before time_limit_s.
    """
    start_time = time.time()
    
    # 1. Initialization
    # Use greedy_balanced_split as a strong starting point
    try:
        partition = tools['greedy_balanced_split'](seed=42)
    except Exception:
        # Fallback to round robin if greedy fails
        partition = tools['round_robin_partition'](seed=42)
        
    # 2. Local Search
    # Iteratively improve by swapping or moving individuals between groups.
    # The tool 'apply_swap_items_across_groups' is designed for this exact purpose,
    # as it respects the 8-group constraint and uses the provided time limit.
    
    # We define a sub-budget for the local search phase
    elapsed = time.time() - start_time
    remaining_time = max(0.1, time_limit_s - elapsed - 0.5)
    
    try:
        improved_partition = tools['apply_swap_items_across_groups'](
            partition=partition, 
            time_limit_s=remaining_time, 
            seed=random.randint(0, 10000)
        )
        partition = improved_partition
    except Exception:
        # If local search fails, return the initial greedy partition
        pass

    # 3. Final Validation
    # Ensure the partition is valid (length, 8 distinct groups)
    is_valid = tools['is_valid_partition'](partition)
    
    if not is_valid:
        # Emergency fallback: ensure 8 groups are present
        assignment = list(range(1, 9)) + [1] * (len(partition) - 8)
        return {'assignment': assignment}

    return {'assignment': partition}