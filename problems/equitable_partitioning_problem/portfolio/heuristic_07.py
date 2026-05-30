# MACE evolved heuristic 07/10 for problem: equitable_partitioning_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the equitable partitioning problem using a multi-start approach.
    By combining different construction heuristics (Greedy and K-Means) 
    with refined local search, it explores a wider solution space than 
    either parent heuristic.
    """
    start_time = time.time()
    
    # Define time segments
    # Overall budget strategy: 
    # 1. Generate multiple candidate solutions using different construction methods.
    # 2. Refine the best candidate using the provided local search.
    
    best_assignment = None
    best_score = float('inf')
    
    # Construction methods to explore
    constructors = [
        lambda: tools['greedy_balanced_split'](seed=random.randint(0, 100000)),
        lambda: tools['balanced_kmeans'](seed=random.randint(0, 100000), max_iters=20),
        lambda: tools['round_robin_partition'](seed=random.randint(0, 100000))
    ]
    
    # Limit construction phase to 30% of time
    construction_deadline = start_time + (time_limit_s * 0.3)
    
    for construct in constructors:
        if time.time() >= construction_deadline:
            break
            
        try:
            candidate = construct()
            score = tools['total_imbalance'](candidate)
            if score < best_score:
                best_score = score
                best_assignment = candidate
        except Exception:
            continue
            
    # Fallback if all constructions failed (should not happen)
    if best_assignment is None:
        best_assignment = tools['round_robin_partition'](seed=42)
        
    # Local search refinement
    # Use remaining time, leaving a 0.2s safety buffer
    remaining_time = time_limit_s - (time.time() - start_time) - 0.2
    
    if remaining_time > 0.1:
        refined_assignment = tools['apply_swap_items_across_groups'](
            partition=best_assignment,
            time_limit_s=remaining_time,
            seed=random.randint(0, 10000)
        )
        # Final check if local search yielded an improvement (it should)
        final_score = tools['total_imbalance'](refined_assignment)
        if final_score < best_score:
            best_assignment = refined_assignment
            
    # Final safety check
    is_ok, _ = tools['is_feasible']({'assignment': best_assignment})
    if not is_ok:
        return {'assignment': tools['round_robin_partition'](seed=123)}
        
    return {'assignment': best_assignment}