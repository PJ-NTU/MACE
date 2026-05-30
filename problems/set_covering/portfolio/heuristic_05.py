# MACE evolved heuristic 05/10 for problem: set_covering
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve the Set Covering Problem using a Greedy Randomized Adaptive Search Procedure (GRASP)
    combined with a local search (redundancy removal).
    """
    start_time = time.time()
    
    # 1. Start with a high-quality greedy solution as a baseline
    best_solution = tools['greedy_cover_by_cost_ratio']()
    best_solution = tools['remove_redundant'](best_solution)
    best_cost = tools['cost_of_selection'](best_solution)
    
    # 2. Iterative improvement (GRASP-like approach)
    # We iteratively relax the solution by removing a random subset of columns,
    # then repairing it greedily and cleaning it.
    while time.time() - start_time < time_limit_s * 0.85:
        # Relax: Remove 20% of columns randomly
        current_selection = list(best_solution)
        if len(current_selection) > 1:
            num_to_remove = max(1, len(current_selection) // 5)
            for _ in range(num_to_remove):
                current_selection.pop(random.randrange(len(current_selection)))
        
        # Repair: Identify uncovered rows and add cheapest covering columns
        uncovered = list(tools['uncovered_rows'](current_selection))
        random.shuffle(uncovered)
        
        for row in uncovered:
            # Find cheapest column covering this row
            candidates = tools['columns_covering_row'](row)
            if not candidates:
                continue # Should not happen if a full cover is possible
            
            best_col = min(candidates, key=lambda c: tools['column_cost'](c))
            current_selection.append(best_col)
            
        # Clean: Remove redundant columns
        current_selection = tools['remove_redundant'](current_selection)
        
        # Update best
        current_cost = tools['cost_of_selection'](current_selection)
        if tools['is_full_cover'](current_selection):
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_selection
                
    # 3. Final polish with the provided default if time permits
    # The default tool often uses ILP/CBC which is excellent for final refinement
    if time.time() - start_time < time_limit_s * 0.95:
        try:
            # Attempt to use the ILP solver as a final optimization step
            # using our best found solution as a hint if possible or just fresh
            ilp_res = tools['ilp_solve_cover'](time_limit_s=max(0.1, time_limit_s * 0.1))
            if ilp_res is not None:
                ilp_sol = tools['remove_redundant'](ilp_res)
                if tools['cost_of_selection'](ilp_sol) < best_cost:
                    best_solution = ilp_sol
        except Exception:
            pass

    return tools['make_solution'](best_solution)