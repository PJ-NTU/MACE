# MACE evolved heuristic 04/10 for problem: set_partitioning
import time

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve a Set Partitioning instance by prioritizing the exact ILP solver
    and implementing adaptive fallback strategies that prioritize the 
    feasibility of the solution under strict time constraints.
    """
    start_time = time.time()
    
    # 1. Primary Strategy: Full ILP
    # The ILP solver is the most robust method for Set Partitioning.
    # We allocate 95% of the time to the primary ILP to maximize the chance of optimality.
    try:
        ilp_time = max(0.5, time_limit_s * 0.95)
        result = tools['ilp_solve_partition'](time_limit_s=ilp_time)
        if result is not None:
            return {"selected_columns": sorted(result)}
    except Exception:
        pass

    # 2. Secondary Strategy: Greedy Construction + ILP Repair
    # If the full ILP fails (e.g., due to complexity or time limit), we use
    # a cost-ratio greedy approach to find a valid base, then refine via ILP.
    try:
        remaining = time_limit_s - (time.time() - start_time)
        if remaining > 0.2:
            greedy_selection = tools['greedy_cover_by_cost_ratio']()
            
            # Check feasibility of greedy result
            is_valid, _ = tools['is_feasible']({"selected_columns": sorted(greedy_selection)})
            if is_valid:
                return {"selected_columns": sorted(greedy_selection)}
            
            # If not feasible, attempt to complete the partial selection via ILP
            # The tool requires a conflict-free selection.
            if tools['is_conflict_free'](greedy_selection):
                completion = tools['complete_partial_via_ilp'](
                    partial=greedy_selection,
                    time_limit_s=max(0.1, remaining - 0.1)
                )
                if completion is not None:
                    return {"selected_columns": sorted(completion)}
    except Exception:
        pass

    # 3. Tertiary Strategy: Minimal Repair
    # If partial completion fails, attempt to build a solution row-by-row
    # using the cheapest column for the first uncovered row.
    try:
        uncovered = list(tools['uncovered_rows']([]))
        if uncovered:
            best_selection = []
            for r in uncovered:
                col = tools['cheapest_column_covering_row'](r, exclude=best_selection)
                if col is not None:
                    best_selection.append(col)
            
            if tools['is_feasible']({"selected_columns": sorted(best_selection)})[0]:
                return {"selected_columns": sorted(best_selection)}
    except Exception:
        pass

    return {"selected_columns": []}