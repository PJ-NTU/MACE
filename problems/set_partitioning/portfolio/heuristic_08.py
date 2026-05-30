# MACE evolved heuristic 08/10 for problem: set_partitioning
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined Set Partitioning solver.
    
    Strategy:
    1. Prioritize the ILP solver as the primary engine, as it is highly optimized
       for set partitioning and handles global constraints better than manual 
       backtracking.
    2. Use a multi-stage approach:
       - First, attempt a full ILP solve with a generous time slice.
       - If that fails or times out, perform a greedy initialization followed by 
         an ILP-based completion.
       - If we have remaining time, attempt a randomized LNS (Large Neighborhood Search)
         to improve the solution cost.
    """
    start_time = time.time()
    
    # Helper to check if we are running out of time
    def time_remaining():
        return time_limit_s - (time.time() - start_time)

    best_sol = None
    best_cost = float('inf')

    # 1. Primary Attempt: Exact ILP solver
    # This covers the widest range of instances effectively.
    try:
        res = tools['ilp_solve_partition'](time_limit_s=min(time_remaining() * 0.7, time_limit_s * 0.5))
        if res is not None:
            best_sol = sorted(res)
            best_cost = tools['cost_of_selection'](best_sol)
    except Exception:
        pass

    # 2. Secondary Attempt: Greedy + Completion
    # If ILP failed, this provides a feasible starting point.
    if best_sol is None and time_remaining() > 0.5:
        try:
            greedy_res = tools['greedy_cover_by_cost_ratio']()
            if tools['is_conflict_free'](greedy_res):
                completed = tools['complete_partial_via_ilp'](
                    partial=greedy_res, 
                    time_limit_s=min(time_remaining() * 0.5, 2.0)
                )
                if completed:
                    best_sol = sorted(completed)
                    best_cost = tools['cost_of_selection'](best_sol)
        except Exception:
            pass

    # 3. LNS Improvement Loop
    # If we have a feasible solution, try to improve it by partially destroying 
    # and re-solving via ILP.
    while best_sol is not None and time_remaining() > 0.5:
        # Destroy: remove 15-25% of columns
        n_remove = max(1, int(len(best_sol) * 0.2))
        indices_to_remove = set(random.sample(best_sol, n_remove))
        partial = [c for c in best_sol if c not in indices_to_remove]
        
        # Repair: use ILP to complete
        repair_time = min(time_remaining() * 0.5, 1.0)
        repaired = tools['complete_partial_via_ilp'](partial, time_limit_s=repair_time)
        
        if repaired is not None:
            new_cost = tools['cost_of_selection'](repaired)
            if new_cost < best_cost:
                best_cost = new_cost
                best_sol = sorted(repaired)
        
        # Prevent infinite loops if no improvement
        if time_remaining() < 0.2:
            break

    # 4. Final Fallback: Simple Greedy if nothing else worked
    if best_sol is None:
        try:
            greedy = tools['greedy_cover_by_cost_ratio']()
            if tools['is_feasible']({"selected_columns": sorted(greedy)})[0]:
                best_sol = sorted(greedy)
        except:
            pass

    return {"selected_columns": best_sol if best_sol is not None else []}