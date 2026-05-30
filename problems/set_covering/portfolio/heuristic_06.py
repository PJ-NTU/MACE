# MACE evolved heuristic 06/10 for problem: set_covering
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized Set Covering solver using a hybrid approach:
    1. Deterministic high-quality greedy start.
    2. ILP solver for global optimization (primary).
    3. Randomized local search (LNS) to refine if time remains.
    """
    start_time = time.time()
    
    # 1. Start with the best known greedy solution.
    # This provides a guaranteed feasible upper bound quickly.
    initial_sol = tools['greedy_cover_by_cost_ratio']()
    best_solution = tools['remove_redundant'](initial_sol)
    best_cost = tools['cost_of_selection'](best_solution)

    # 2. Primary Strategy: ILP
    # The ILP solver is the most effective tool for Set Covering.
    # We allocate the majority of the time budget to it.
    ilp_time_limit = time_limit_s * 0.7
    ilp_sol = tools['ilp_solve_cover'](time_limit_s=ilp_time_limit)
    
    if ilp_sol is not None:
        refined_ilp = tools['remove_redundant'](ilp_sol)
        if tools['cost_of_selection'](refined_ilp) < best_cost:
            best_solution = refined_ilp
            best_cost = tools['cost_of_selection'](best_solution)

    # 3. Secondary Strategy: Large Neighborhood Search (LNS)
    # If time remains, perform targeted destruction and greedy repair.
    # This helps escape local optima if the ILP didn't finish perfectly.
    while time.time() - start_time < time_limit_s * 0.95:
        if len(best_solution) <= 1:
            break
            
        # Destruction: Remove a chunk of the current solution
        temp_sol = list(best_solution)
        random.shuffle(temp_sol)
        remove_count = max(1, len(temp_sol) // 4)
        del temp_sol[:remove_count]
        
        # Repair: Greedy repair using cost-effectiveness ratio
        uncovered = list(tools['uncovered_rows'](temp_sol))
        random.shuffle(uncovered)
        
        repaired = True
        for r in uncovered:
            candidates = tools['columns_covering_row'](r)
            if not candidates:
                repaired = False
                break
            # Pick best by cost/new_coverage ratio
            best_c = min(candidates, key=lambda c: tools['column_cost'](c) / 
                         max(1, len(tools['column_covers'](c).intersection(set(uncovered)))))
            temp_sol.append(best_c)
        
        if repaired:
            temp_sol = tools['remove_redundant'](temp_sol)
            curr_cost = tools['cost_of_selection'](temp_sol)
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_solution = temp_sol
        
        # Break if we are close to time limit
        if time.time() - start_time > time_limit_s * 0.9:
            break

    return tools['make_solution'](best_solution)