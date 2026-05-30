# MACE evolved heuristic 03/10 for problem: set_partitioning
import time

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Set Partitioning problem using a combination of the built-in 
    ILP solver (as the primary strategy) and a fallback greedy construction 
    if the ILP fails or times out.
    """
    start_time = time.time()
    num_rows = instance["num_rows"]
    
    # 1. Attempt exact solution via ILP as the primary strategy.
    # The ILP solver is generally the most efficient approach for Set Partitioning.
    try:
        # Give the ILP solver most of the available time.
        ilp_time_limit = max(1.0, time_limit_s * 0.9)
        result = tools['ilp_solve_partition'](time_limit_s=ilp_time_limit)
        
        if result is not None:
            # Sort the result as required by the interface.
            return {"selected_columns": sorted(result)}
    except Exception:
        # If the ILP solver fails or errors out, proceed to fallback.
        pass

    # 2. Fallback: Greedy construction + completion.
    # If ILP didn't return a result, try a greedy approach.
    try:
        # Greedy initialization
        greedy_selection = tools['greedy_cover_by_cost_ratio']()
        
        # Check if greedy already solved it
        if tools['is_feasible']({"selected_columns": sorted(greedy_selection)})[0]:
            return {"selected_columns": sorted(greedy_selection)}
        
        # If partial, try to complete via ILP.
        # Ensure we don't exceed the time limit.
        remaining_time = time_limit_s - (time.time() - start_time)
        if remaining_time > 0.5:
            completion = tools['complete_partial_via_ilp'](
                partial=greedy_selection, 
                time_limit_s=remaining_time
            )
            if completion is not None:
                return {"selected_columns": sorted(completion)}
                
    except Exception:
        pass

    # 3. Final safety fallback: return empty if absolutely nothing found.
    # The evaluation function will reject this, but it fulfills the interface.
    return {"selected_columns": []}