# MACE evolved heuristic 01/10 for problem: set_partitioning
import time

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the set partitioning problem using a hybrid approach:
    1. Attempts an exact ILP solve if time permits.
    2. Falls back to a greedy construction completed by ILP if the ILP solve
       is too slow or fails to find a solution within the time limit.
    """
    start_time = time.time()
    
    # 1. Try to solve using the built-in ILP solver
    # We allocate most of the time to the ILP solver.
    ilp_time_limit = max(1.0, time_limit_s * 0.8)
    
    try:
        solution = tools['ilp_solve_partition'](time_limit_s=ilp_time_limit)
        if solution is not None:
            return {"selected_columns": sorted(solution)}
    except Exception:
        pass

    # 2. Fallback: Greedy construction + ILP completion
    # If ILP fails, use the greedy heuristic provided by tools
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time < 0.1:
        return {"selected_columns": []}
        
    try:
        greedy_selection = tools['greedy_cover_by_cost_ratio']()
        
        # If greedy selection isn't already a full partition, attempt to fill it
        if not tools['is_conflict_free'](greedy_selection):
            # Try to fix by stripping conflicts or just restart
            greedy_selection = []
            
        # Try to complete the greedy selection using ILP
        completed = tools['complete_partial_via_ilp'](
            partial=greedy_selection, 
            time_limit_s=min(remaining_time, 2.0)
        )
        
        if completed is not None:
            return {"selected_columns": sorted(completed)}
            
    except Exception:
        pass

    # 3. Last resort: Return a known feasible solution if possible, 
    # or empty if all else fails.
    return {"selected_columns": []}