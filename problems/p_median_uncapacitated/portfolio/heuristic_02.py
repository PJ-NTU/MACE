# MACE evolved heuristic 02/10 for problem: p_median_uncapacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the uncapacitated p-median problem using a greedy construction
    followed by a Teitz-Bart (interchange) local search.
    """
    start_time = time.time()
    
    # Extract dimensions
    n = instance['n']
    p = instance['p']
    
    # 1. Greedy construction
    # We use the provided greedy tool if available, otherwise implement a basic 
    # greedy start to ensure we have a valid initial set of p medians.
    try:
        current_medians = tools['greedy_add_one_until_p']()
    except (KeyError, Exception):
        # Fallback greedy construction
        current_medians = []
        remaining = list(range(1, n + 1))
        while len(current_medians) < p:
            best_m = -1
            best_cost = float('inf')
            for m in remaining:
                candidate = current_medians + [m]
                cost = tools['cost_given_open'](candidate)
                if cost < best_cost:
                    best_cost = cost
                    best_m = m
            current_medians.append(best_m)
            remaining.remove(best_m)

    # 2. Local Search (Teitz-Bart Interchange)
    # We use the provided tool which is optimized for time monitoring.
    # We allow the tool to run until it converges or the time limit is reached.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        try:
            # Using apply_swap_one_for_one as it is robust for p-median
            optimized_medians = tools['apply_swap_one_for_one'](
                current_medians, 
                time_limit_s=remaining_time, 
                first_improvement=True
            )
            current_medians = optimized_medians
        except (KeyError, Exception):
            # If tool fails, proceed with the greedy result
            pass

    # Ensure the final result adheres to the required format
    # The tools guarantee a list of size p, but we verify just in case.
    if len(current_medians) != p:
        # Fallback to a simple valid set if something went wrong
        current_medians = list(range(1, p + 1))

    return {"medians": list(current_medians)}