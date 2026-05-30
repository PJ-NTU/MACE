# MACE evolved heuristic 04/10 for problem: uncapacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Uncapacitated Warehouse Location Problem using a hybrid approach:
    1. Initial solutions via Greedy Add and Greedy Drop heuristics.
    2. Local search via Swap (Open/Close) and single-warehouse toggle improvements.
    3. Final refinement using ILP if time permits.
    """
    start_time = time.time()

    def get_time_left():
        return time_limit_s - (time.time() - start_time)

    best_sol = None
    best_cost = float('inf')

    # Candidate open sets to explore
    candidates = []
    
    # Heuristic 1: Greedy Add
    if get_time_left() > 0.1:
        add_set = tools['greedy_add_one'](time_limit_s=max(0.1, get_time_left() * 0.2))
        candidates.append(add_set)
    
    # Heuristic 2: Greedy Drop
    if get_time_left() > 0.1:
        drop_set = tools['greedy_drop_one'](time_limit_s=max(0.1, get_time_left() * 0.2))
        candidates.append(drop_set)

    # Local Search: Swap refinement
    for cand in candidates:
        if get_time_left() < 0.1:
            break
        refined_set = tools['apply_swap_open_close'](cand, time_limit_s=max(0.1, get_time_left() * 0.2))
        cost = tools['cost_given_open'](refined_set)
        if cost < best_cost:
            best_cost = cost
            best_sol = tools['solution_from_open'](refined_set)

    # ILP Refinement: Use ILP to find the global optimum if time allows
    # We use the best found heuristic solution as a warm start if possible,
    # but the provided ILP tool doesn't take an incumbent, so we run it directly.
    if get_time_left() > 1.0:
        ilp_sol = tools['ilp_uwl'](time_limit_s=max(0.5, get_time_left() - 0.2))
        if ilp_sol is not None:
            # Re-calculate objective to ensure consistency
            cost = tools['objective'](ilp_sol)
            if cost < best_cost:
                best_sol = ilp_sol
                best_cost = cost

    # Fallback: If no solutions found, use greedy add
    if best_sol is None:
        final_set = tools['greedy_add_one'](time_limit_s=max(0.1, get_time_left()))
        best_sol = tools['solution_from_open'](final_set)

    return best_sol