# MACE evolved heuristic 06/10 for problem: set_partitioning
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for Set Partitioning.
    
    Hypothesis:
    - Density = (Sum of rows covered by all columns) / (num_rows * num_columns)
    - Low-density instances are often sparse and well-structured, making them 
      suitable for exact methods or simple greedy-repair approaches (Parent A).
    - High-density instances have many overlapping constraints, making them 
      'tight' and more suitable for constraint-based backtracking (Parent B).
    """
    start_time = time.time()
    
    num_rows = instance.get("num_rows", 0)
    num_columns = instance.get("num_columns", 0)
    columns_info = instance.get("columns_info", {})
    
    # Calculate density
    total_cover_slots = sum(len(rows) for _, rows in columns_info.values())
    density = total_cover_slots / (num_rows * num_columns) if num_rows * num_columns > 0 else 0
    
    # Heuristic: If sparse, use LNS (Parent A). If dense, use Backtracking (Parent B).
    # Threshold 0.05 determined as a balance point for constraint satisfaction difficulty.
    if density < 0.05:
        # Parent A: Greedy + LNS
        greedy_partial = tools['greedy_cover_by_cost_ratio']()
        best_sol = tools['complete_partial_via_ilp'](greedy_partial, time_limit_s=min(2.0, time_limit_s * 0.3))
        
        if best_sol is None:
            best_sol = tools['ilp_solve_partition'](time_limit_s=min(5.0, time_limit_s * 0.5))
        
        if best_sol is not None:
            best_sol = sorted(best_sol)
            best_cost = tools['cost_of_selection'](best_sol)
            
            while time.time() - start_time < time_limit_s * 0.9:
                n_remove = max(1, int(len(best_sol) * 0.15))
                indices_to_remove = set(random.sample(best_sol, n_remove))
                partial = [c for c in best_sol if c not in indices_to_remove]
                
                repair_time = min(1.0, (time_limit_s - (time.time() - start_time)) * 0.5)
                repaired = tools['complete_partial_via_ilp'](partial, time_limit_s=repair_time)
                
                if repaired is not None:
                    new_cost = tools['cost_of_selection'](repaired)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_sol = sorted(repaired)
            return {"selected_columns": best_sol}
    
    else:
        # Parent B: Backtracking + ILP fallback
        row_to_cols = {r: tools['columns_covering_row'](r) for r in range(1, num_rows + 1)}
        best_solution = None
        min_cost = float('inf')
        
        def backtrack(current_selection, current_selection_rows, uncovered_rows, current_cost):
            nonlocal best_solution, min_cost
            if time.time() - start_time > time_limit_s * 0.9: return
            if not uncovered_rows:
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_solution = sorted(current_selection)
                return
            if current_cost >= min_cost: return
            
            row = min(uncovered_rows, key=lambda r: len([c for c in row_to_cols[r] if tools['column_rows'](c).isdisjoint(current_selection_rows)]))
            candidates = sorted([c for c in row_to_cols[row] if tools['column_rows'](c).isdisjoint(current_selection_rows)], key=lambda c: tools['column_cost'](c))
            
            for col in candidates:
                col_rows = tools['column_rows'](col)
                current_selection.append(col)
                current_selection_rows.update(col_rows)
                backtrack(current_selection, current_selection_rows, uncovered_rows - col_rows, current_cost + tools['column_cost'](col))
                current_selection.pop()
                current_selection_rows.difference_update(col_rows)
                if best_solution and min_cost == 0: break

        backtrack([], set(), set(range(1, num_rows + 1)), 0)
        
        if best_solution:
            return {"selected_columns": best_solution}
            
    # Final fallback: Full ILP
    ilp_sol = tools['ilp_solve_partition'](time_limit_s=max(1.0, time_limit_s * 0.2))
    return {"selected_columns": sorted(ilp_sol)} if ilp_sol else {"selected_columns": []}