# MACE evolved heuristic 06/10 for problem: equitable_partitioning_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Advanced heuristic for equitable partitioning:
    1. Uses ILP if the instance size is small enough (high precision).
    2. Uses balanced_kmeans for medium/large instances to get a better 
       starting point than greedy_balanced_split.
    3. Performs iterative refinement via apply_swap_items_across_groups.
    """
    start_time = time.time()
    n_individuals = tools['n_items']()
    
    # 1. Selection of Construction Heuristic based on problem scale
    # ILP is very strong for small instances (n <= 80).
    # balanced_kmeans provides a better spatial start for larger instances than greedy_balanced_split.
    
    assignment = None
    
    # Strategy: Try ILP if likely to be fast, otherwise use kmeans.
    if n_individuals <= 70 and (time_limit_s > 5.0):
        try:
            assignment = tools['ilp_partition'](time_limit_s=min(15.0, time_limit_s * 0.4))
        except:
            assignment = None
            
    if assignment is None:
        try:
            # balanced_kmeans is generally superior to greedy for L1-imbalance 
            # because it optimizes the attribute space centroids.
            assignment = tools['balanced_kmeans'](seed=random.randint(0, 10000), max_iters=50)
        except:
            assignment = tools['greedy_balanced_split'](seed=random.randint(0, 10000))
            
    # 2. Iterative Refinement
    # Use the remaining time budget to perform local search.
    # We leave a buffer of 0.2s for final processing.
    remaining = time_limit_s - (time.time() - start_time) - 0.2
    
    if remaining > 0.1:
        try:
            assignment = tools['apply_swap_items_across_groups'](
                partition=assignment,
                time_limit_s=remaining,
                seed=random.randint(0, 10000)
            )
        except Exception:
            pass

    # 3. Final Validation and Safety
    # Ensure the returned solution is strictly feasible.
    feasible, _ = tools['is_feasible']({"assignment": assignment})
    if not feasible:
        # Fallback to a guaranteed valid round-robin partition
        assignment = tools['round_robin_partition'](seed=42)
        
    return {"assignment": assignment}