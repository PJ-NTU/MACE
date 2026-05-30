# MACE evolved heuristic 10/10 for problem: set_partitioning
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined Set Partitioning solver using a combination of a robust greedy
    initialization and iterative local search (LNS) driven by the ILP solver.
    """
    start_time = time.time()
    
    # 1. Initialization: Get a solid baseline
    # Greedy cover provides a good starting point for LNS
    current_best = tools['greedy_cover_by_cost_ratio']()
    
    # If greedy fails to cover everything, attempt to complete it via ILP
    if tools['uncovered_rows'](current_best):
        completion = tools['complete_partial_via_ilp'](current_best, time_limit_s=min(2.0, time_limit_s * 0.2))
        if completion:
            current_best = completion
    
    # 2. Safety fallback: If still no full solution, use the full ILP solver
    if not current_best or tools['uncovered_rows'](current_best):
        ilp_res = tools['ilp_solve_partition'](time_limit_s=max(1.0, time_limit_s * 0.5))
        if ilp_res:
            current_best = sorted(ilp_res)
        else:
            return {"selected_columns": []}

    # 3. Iterative Improvement (LNS)
    # Focus on removing expensive columns and re-optimizing with ILP
    best_cost = tools['cost_of_selection'](current_best)
    
    while time.time() - start_time < time_limit_s * 0.9:
        # Determine number of columns to destroy (dynamic size)
        # We target removing 10-25% of the selection to create a manageable subproblem
        num_to_destroy = max(1, int(len(current_best) * 0.2))
        
        # Destroy phase: Remove random columns
        removed = random.sample(current_best, num_to_destroy)
        partial = [c for c in current_best if c not in removed]
        
        # Repair phase: Use ILP to find the cheapest way to cover the uncovered rows
        # The ILP solver handles the logic of finding the best fill-in columns
        repair_time = min(2.0, (time_limit_s - (time.time() - start_time)) * 0.5)
        repaired = tools['complete_partial_via_ilp'](partial, time_limit_s=repair_time)
        
        if repaired:
            new_cost = tools['cost_of_selection'](repaired)
            if new_cost < best_cost:
                best_cost = new_cost
                current_best = sorted(repaired)
        
        # Periodically break if we have a very good solution (or hit time limit)
        if time.time() - start_time > time_limit_s * 0.95:
            break
            
    return {"selected_columns": sorted(current_best)}