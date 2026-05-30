# MACE evolved heuristic 04/10 for problem: p_median_uncapacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Redesigned solver for the Uncapacitated P-Median Problem.
    
    Diagnosis of parent:
    1. Greedy initialization is deterministic and prone to trapping in poor local optima.
    2. Pure Teitz-Bart (1-for-1) is often insufficient for larger instances.
    3. The lack of multi-start or randomized restarts limits exploration.
    
    Redesign:
    1. Multi-start approach: Perform a randomized greedy construction followed by 
       the robust apply_interchange_LK (Lin-Kernighan style) for deeper local exploration.
    2. Time-aware budget management: Allocate time across multiple restarts.
    3. Use of best-performing local search primitives provided by tools.
    """
    start_time = time.time()
    n = instance['n']
    p = instance['p']
    
    best_overall_medians = None
    best_overall_cost = float('inf')
    
    # Heuristic: Allow for multiple restarts if time permits.
    # Start with the greedy construction, then follow with a randomized variant.
    num_restarts = 0
    while time.time() - start_time < time_limit_s * 0.85:
        num_restarts += 1
        
        if num_restarts == 1:
            # First pass: deterministic greedy
            current_medians = tools['greedy_add_one_until_p']()
        else:
            # Subsequent passes: Randomized construction
            # Pick a random starting point and fill greedily
            pool = list(range(1, n + 1))
            random.shuffle(pool)
            current_medians = pool[:p]
            
        # Refinement: Apply Lin-Kernighan flavor interchange (best-improvement)
        # This is generally more robust than standard 1-for-1 swap.
        remaining_time = time_limit_s - (time.time() - start_time)
        if remaining_time < 0.1:
            break
            
        optimized_medians = tools['apply_interchange_LK'](
            current_medians, 
            time_limit_s=remaining_time
        )
        
        # Evaluate
        current_cost = tools['cost_given_open'](optimized_medians)
        
        if current_cost < best_overall_cost:
            best_overall_cost = current_cost
            best_overall_medians = optimized_medians
            
        # If we have a very small n, one pass is likely optimal.
        if n < 50:
            break

    # Final fallback if something went wrong
    if best_overall_medians is None:
        best_overall_medians = tools['greedy_add_one_until_p']()

    return {"medians": list(best_overall_medians)}