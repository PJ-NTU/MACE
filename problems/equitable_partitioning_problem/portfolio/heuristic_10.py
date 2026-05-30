# MACE evolved heuristic 10/10 for problem: equitable_partitioning_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved solver for the equitable partitioning problem.
    Modification: Enhanced the initialization phase by incorporating a multi-start 
    strategy for the greedy construction. Instead of relying on a single seed, 
    we evaluate multiple greedy starts within the time budget to select the best 
    starting point for the local search, improving the likelihood of finding a 
    superior global basin.
    """
    start_time = time.time()
    n_items = len(instance['data'])
    
    # 1. Multi-start construction: Evaluate multiple greedy seeds to find the best start.
    best_assignment = None
    best_score = float('inf')
    
    # Allocate a portion of time for construction
    construction_time_limit = time_limit_s * 0.2
    seeds = [42, 123, 7, 999, 2024]
    
    for seed in seeds:
        if (time.time() - start_time) > construction_time_limit:
            break
        
        # Fallback to ILP if very small, otherwise greedy multi-start
        if n_items <= 100 and best_assignment is None:
            assignment = tools['ilp_partition'](time_limit_s=min(time_limit_s * 0.2, 2.0))
            if assignment:
                best_assignment = assignment
                best_score = tools['total_imbalance'](assignment)
                break
        
        current_assignment = tools['greedy_balanced_split'](seed=seed)
        current_score = tools['total_imbalance'](current_assignment)
        
        if current_score < best_score:
            best_score = current_score
            best_assignment = current_assignment
            
    if best_assignment is None:
        best_assignment = tools['round_robin_partition'](seed=0)
        best_score = tools['total_imbalance'](best_assignment)
        
    # 2. Main Refinement loop
    # Use the strong local search tool provided in the interface.
    remaining_time = time_limit_s - (time.time() - start_time)
    
    if remaining_time > 0.5:
        refined = tools['apply_swap_items_across_groups'](
            partition=list(best_assignment),
            time_limit_s=remaining_time * 0.9,
            seed=random.randint(0, 10000)
        )
        refined_score = tools['total_imbalance'](refined)
        if refined_score < best_score:
            best_assignment = refined

    # 3. Final check to ensure we provide a valid solution.
    is_ok, _ = tools['is_feasible']({'assignment': best_assignment})
    if not is_ok:
        best_assignment = tools['round_robin_partition'](seed=0)
        
    return {'assignment': best_assignment}