# MACE evolved heuristic 02/10 for problem: set_partitioning
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve a Set Partitioning instance using an ILP-based approach.
    Given the availability of `ilp_solve_partition` and `complete_partial_via_ilp`,
    these are the most robust tools for this NP-hard problem.
    """
    start_time = time.time()
    
    # 1. Attempt to solve the full problem using the internal ILP solver.
    # This is the gold standard if the problem size allows it within the time limit.
    try:
        # We allocate 80% of the time limit to the initial ILP call.
        res = tools['ilp_solve_partition'](time_limit_s=time_limit_s * 0.8)
        if res is not None:
            return {"selected_columns": sorted(res)}
    except Exception:
        pass

    # 2. If ILP fails or times out, try a greedy construction followed by completion.
    # Greedy heuristic: picking cost-effective columns.
    try:
        if time.time() - start_time < time_limit_s * 0.9:
            greedy_res = tools['greedy_cover_by_cost_ratio']()
            
            # Check if it's already a valid partition
            sol = {"selected_columns": sorted(greedy_res)}
            is_f, _ = tools['is_feasible'](sol)
            if is_f:
                return sol
            
            # If not, try to complete the partial selection using ILP.
            # We must ensure the input to complete_partial_via_ilp is conflict-free.
            if tools['is_conflict_free'](greedy_res):
                completed = tools['complete_partial_via_ilp'](
                    partial=greedy_res, 
                    time_limit_s=(time_limit_s - (time.time() - start_time)) * 0.5
                )
                if completed:
                    return {"selected_columns": sorted(completed)}
    except Exception:
        pass

    # 3. Fallback: If no other method works, try to find a feasible solution 
    # by iteratively picking the cheapest column for each row until full.
    # This is a last-resort heuristic.
    try:
        current_selection = []
        rows_to_cover = list(range(1, instance['num_rows'] + 1))
        random.shuffle(rows_to_cover)
        
        for r in rows_to_cover:
            if r not in tools['covered_rows'](current_selection):
                best_col = tools['cheapest_column_covering_row'](r, exclude=current_selection)
                if best_col is not None:
                    # Only add if it doesn't create a conflict
                    new_selection = current_selection + [best_col]
                    if tools['is_conflict_free'](new_selection):
                        current_selection = new_selection
        
        # Final check
        sol = {"selected_columns": sorted(current_selection)}
        is_f, _ = tools['is_feasible'](sol)
        if is_f:
            return sol
    except Exception:
        pass

    # Return the best we could do (even if empty, as per template)
    return {"selected_columns": []}