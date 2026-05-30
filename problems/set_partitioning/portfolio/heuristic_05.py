# MACE evolved heuristic 05/10 for problem: set_partitioning
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Fixed version of the backtracking heuristic.
    Ensures that if the backtracking search fails to find a solution, 
    it falls back to a reliable ILP-based solver to guarantee feasibility.
    """
    start_time = time.time()
    num_rows = instance["num_rows"]
    
    # Pre-compute row-to-columns mapping for fast lookup
    row_to_cols = {r: tools['columns_covering_row'](r) for r in range(1, num_rows + 1)}
    
    best_solution = None
    min_cost = float('inf')
    
    def backtrack(current_selection, current_selection_rows, uncovered_rows, current_cost):
        nonlocal best_solution, min_cost
        
        # Check time limit
        if time.time() - start_time > time_limit_s * 0.95:
            return

        # If no more rows, we found a feasible solution
        if not uncovered_rows:
            if current_cost < min_cost:
                min_cost = current_cost
                best_solution = sorted(current_selection)
            return
        
        # Pruning
        if current_cost >= min_cost:
            return

        # Pick a row (MRV: row with fewest remaining options)
        row = min(uncovered_rows, key=lambda r: len([c for c in row_to_cols[r] if tools['column_rows'](c).isdisjoint(current_selection_rows)]))
        
        # Try columns that cover this row and don't conflict
        candidates = [c for c in row_to_cols[row] if tools['column_rows'](c).isdisjoint(current_selection_rows)]
        # Sort candidates by cost
        candidates.sort(key=lambda c: tools['column_cost'](c))
        
        for col in candidates:
            col_rows = tools['column_rows'](col)
            
            # Apply move
            current_selection.append(col)
            current_selection_rows.update(col_rows)
            
            backtrack(current_selection, current_selection_rows, uncovered_rows - col_rows, current_cost + tools['column_cost'](col))
            
            # Backtrack
            current_selection.pop()
            current_selection_rows.difference_update(col_rows)
            
            if best_solution and min_cost == 0: break

    # State for recursion
    backtrack([], set(), set(range(1, num_rows + 1)), 0)
    
    if best_solution:
        return {"selected_columns": best_solution}
    
    # Fallback to ILP if backtracking failed to find a valid solution
    ilp_sol = tools['ilp_solve_partition'](time_limit_s=(time_limit_s - (time.time() - start_time)))
    if ilp_sol is not None:
        return {"selected_columns": sorted(ilp_sol)}
        
    return {"selected_columns": []}