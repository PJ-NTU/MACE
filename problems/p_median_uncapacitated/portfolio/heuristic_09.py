# MACE evolved heuristic 09/10 for problem: p_median_uncapacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized solver for the Uncapacitated P-Median Problem.
    
    The strategy employs a robust multi-start framework:
    1. Construction: Uses the greedy initialization (deterministic) followed by 
       randomized shuffles to explore the solution space.
    2. Refinement: Utilizes the provided high-performance 'apply_interchange_LK' 
       local search tool, which is superior to manual swap implementations.
    3. Time-Awareness: Dynamically adjusts iterations based on the available
       time budget to maximize the quality of the final incumbent.
    """
    start_time = time.time()
    n = instance['n']
    p = instance['p']
    
    best_medians = None
    best_cost = float('inf')
    
    # 1. Deterministic Greedy Start (Strong Baseline)
    try:
        current_medians = tools['greedy_add_one_until_p']()
        cost = tools['cost_given_open'](current_medians)
        best_medians = current_medians
        best_cost = cost
    except:
        pass

    # 2. Iterative Randomized Multi-start
    # We use a loop that continues as long as we have significant time remaining.
    # We favor LK interchange as it is the most robust tool provided.
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.9:
        iteration += 1
        
        # Construct a random candidate
        # Using a mix of random selection to jump to unexplored regions
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        initial_guess = candidates[:p]
        
        # Apply the high-quality interchange local search
        remaining_time = time_limit_s - (time.time() - start_time)
        if remaining_time < 0.05:
            break
            
        try:
            optimized = tools['apply_interchange_LK'](
                initial_guess, 
                time_limit_s=remaining_time
            )
            
            # Re-verify feasibility and cost
            is_ok, _ = tools['validate_open_count'](optimized)
            if is_ok:
                current_cost = tools['cost_given_open'](optimized)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_medians = optimized
        except:
            continue
            
        # If we have a very small instance, we likely hit global optimum quickly
        if n <= 20:
            break

    # Final fallback if something went wrong
    if best_medians is None or len(best_medians) != p:
        best_medians = tools['greedy_add_one_until_p']()

    return {"medians": list(best_medians)}