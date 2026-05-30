# MACE evolved heuristic 03/10 for problem: p_median_uncapacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved heuristic for the Uncapacitated p-median problem.
    
    Strategy:
    1. Start with the strong greedy construction.
    2. Perform a best-improvement local search (Teitz-Bart) using 
       apply_interchange_LK, which is more robust against local optima 
       than simple first-improvement.
    3. Spend the remaining time budget on a multi-start random perturbation 
       to escape local optima if the primary search completes early.
    """
    start_time = time.time()
    n = instance['n']
    p = instance['p']
    
    def get_time_left():
        return max(0.0, time_limit_s - (time.time() - start_time))

    # 1. Primary Construction
    best_medians = tools['greedy_add_one_until_p']()
    best_cost = tools['objective']({"medians": best_medians})

    # 2. Primary Refinement (Lin-Kernighan flavor)
    # Using the more robust LK tool for the bulk of the time budget
    time_for_lk = get_time_left() * 0.7
    if time_for_lk > 0.5:
        refined = tools['apply_interchange_LK'](
            open_set=best_medians, 
            time_limit_s=time_for_lk
        )
        current_cost = tools['objective']({"medians": refined})
        if current_cost < best_cost:
            best_medians = refined
            best_cost = current_cost

    # 3. Escaping local optima: Multi-start perturbation
    # If time remains, perform random swaps to explore different basins
    while get_time_left() > 0.5:
        # Create a candidate by perturbing the current best
        candidate = list(best_medians)
        # Swap 1 random median out
        idx_to_remove = random.randrange(p)
        candidate.pop(idx_to_remove)
        
        # Add 1 random node not in current set
        current_set = set(candidate)
        candidates = [i for i in range(1, n + 1) if i not in current_set]
        if not candidates:
            break
        candidate.append(random.choice(candidates))
        
        # Refine the perturbed candidate
        try:
            refined = tools['apply_swap_one_for_one'](
                open_set=candidate,
                time_limit_s=min(0.5, get_time_left()),
                first_improvement=True
            )
            c_cost = tools['objective']({"medians": refined})
            if c_cost < best_cost:
                best_medians = refined
                best_cost = c_cost
        except:
            pass

    return {"medians": best_medians}