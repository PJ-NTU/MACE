# MACE evolved heuristic 01/10 for problem: p_median_uncapacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the uncapacitated p-median problem using a robust
    Greedy Construction followed by a Teitz-Bart Local Search.
    """
    start_time = time.time()
    
    # 1. Greedy construction to get a strong initial solution
    # The tools provide a greedy heuristic that works well for UPM.
    try:
        initial_medians = tools['greedy_add_one_until_p']()
    except Exception:
        # Fallback to random if tool fails
        n = instance['n']
        p = instance['p']
        initial_medians = random.sample(range(1, n + 1), p)

    # 2. Local Search (Teitz-Bart / Interchange)
    # This refines the initial solution by swapping medians for non-medians.
    # We use the provided tool which handles time monitoring internally.
    
    # Calculate remaining time for the local search phase
    elapsed = time.time() - start_time
    remaining = max(0.1, time_limit_s - elapsed - 0.5)  # Buffer 0.5s for overhead
    
    try:
        refined_medians = tools['apply_swap_one_for_one'](
            open_set=initial_medians, 
            time_limit_s=remaining, 
            first_improvement=True
        )
    except Exception:
        refined_medians = initial_medians

    # 3. Final validation and return
    # Ensure the result meets the specific return format of the problem
    solution = {"medians": list(refined_medians)}
    
    # If for some reason the tools returned an invalid size, fix it locally
    if len(solution["medians"]) != instance['p']:
        # Fallback to simple random if something went wrong
        solution["medians"] = random.sample(range(1, instance['n'] + 1), instance['p'])
        
    return solution