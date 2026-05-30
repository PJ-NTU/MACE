# MACE evolved heuristic 02/10 for problem: capacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Implements a hybrid approach:
    1. Uses the provided ILP solver (python-mip via tools) for an optimal single-source baseline.
    2. Since the problem allows splittable demand, the single-source solution is a valid 
       subset of the feasible space.
    3. If the ILP fails or time is tight, falls back to a greedy construction heuristic.
    """
    start_time = time.time()

    # 1. Attempt to get an optimal single-source solution using the provided ILP tool.
    # The single-source variant is a strong heuristic for the splittable version.
    try:
        ilp_time = min(time_limit_s * 0.7, 5.0)
        ilp_sol = tools['ilp_cwl'](time_limit_s=ilp_time)
        if ilp_sol:
            # Convert single-source assignments to the splittable format required
            m = instance['m']
            n = instance['n']
            warehouse_open = ilp_sol['warehouse_open']
            # ILP returns 'assignments' as a list of warehouse indices or similar.
            # We need to map the single-source assignment to the n x m matrix.
            raw_assignments = ilp_sol.get('assignments', [])
            
            # The ilp_cwl tool returns a full dict. If it's already in the format
            # expected by the evaluator, we can return it.
            return ilp_sol
    except Exception:
        pass

    # 2. Fallback / Refinement: Greedy construction
    # Use the density-based greedy heuristic provided by tools.
    open_set, assignment = tools['greedy_open_by_density']()
    
    # Simple Local Search: Try to swap opened/closed warehouses
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        open_set, assignment = tools['apply_swap_open_close'](open_set, time_limit_s=remaining_time)

    # 3. Convert to full splittable format
    # The assignment returned by tools is single-source (list of indices).
    # We convert to n x m matrix where demand is placed entirely on the chosen warehouse.
    n = instance['n']
    m = instance['m']
    customers = instance['customers']
    
    assignments = [[0.0 for _ in range(m)] for _ in range(n)]
    for j in range(n):
        wh_idx = assignment[j]
        if wh_idx != -1:
            assignments[j][wh_idx] = float(customers[j]['demand'])
    
    # Build solution dict
    sol = {
        'warehouse_open': open_set,
        'assignments': assignments
    }
    
    # Calculate objective via eval_func indirectly if possible,
    # or rely on the fact that tools['to_solution'] handles the structure.
    # However, since we built it manually, we ensure it's valid.
    final_sol = tools['to_solution'](open_set, assignment)
    
    # Verify feasibility before returning
    is_f, msg = tools['is_feasible'](final_sol)
    if is_f:
        return final_sol
    
    # If not feasible, return an empty dict or the last best effort
    return final_sol if is_f else {}