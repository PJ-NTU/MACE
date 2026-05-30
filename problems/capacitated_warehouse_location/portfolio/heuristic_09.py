# MACE evolved heuristic 09/10 for problem: capacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized solver for the Capacitated Warehouse Location Problem.
    
    Strategy:
    1. Prioritize ILP (Exact Solver) as it excels at finding global optima for small/medium instances.
    2. Implement a robust fallback: If ILP fails, use a density-based construction 
       followed by an iterative improvement loop that attempts to fix unassigned 
       customers by greedily opening the best-value remaining warehouses.
    3. Self-monitor time throughout the execution to ensure a feasible solution 
       is returned.
    """
    start_time = time.time()
    
    # 1. Primary: Exact solver (ILP)
    # The ILP solver provides the best performance for the benchmark constraints.
    # We allocate up to 80% of the time limit to the exact solver.
    ilp_time = min(time_limit_s * 0.8, 10.0)
    ilp_res = tools['ilp_cwl'](time_limit_s=ilp_time)
    if ilp_res is not None:
        return ilp_res

    # 2. Fallback: Construction Heuristic
    # Start with a density-based baseline.
    open_set, assignment = tools['greedy_open_by_density']()
    
    # 3. Repair/Refinement Loop
    # If the assignment is incomplete, try to iteratively add the most cost-effective 
    # unopened warehouse to accommodate remaining demand.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        # First, try to fix unassigned customers by brute-forcing opening 
        # additional warehouses if the initial density heuristic failed.
        unassigned = tools['unassigned_customers'](assignment)
        if unassigned:
            # Try to open all warehouses not yet in set to guarantee feasibility
            # if we have time.
            all_warehouses = set(range(tools['n_warehouses']()))
            open_set = list(all_warehouses)
            assignment = tools['greedy_serve_nearest'](open_set)
        
        # Apply local search to improve the configuration
        open_set, assignment = tools['apply_swap_open_close'](
            open_set, 
            time_limit_s=min(remaining_time, 2.0)
        )

    # 4. Final Conversion
    # Ensure the output is in the required splittable format.
    m = tools['n_warehouses']()
    n = tools['n_customers']()
    
    warehouse_open = [0] * m
    for i in open_set:
        warehouse_open[i] = 1
        
    assignments = [[0.0 for _ in range(m)] for _ in range(n)]
    for j in range(n):
        wh_idx = assignment[j]
        if wh_idx != -1:
            assignments[j][wh_idx] = tools['customer_demand'](j)
            
    # Compute final objective
    # Note: tools['to_solution'] is robust for building the final dict
    # but we must ensure it matches the structure expected by the evaluator.
    sol = tools['to_solution'](open_set, assignment)
    
    # Final safety check
    is_f, _ = tools['is_feasible'](sol)
    if not is_f:
        # Emergency fallback: open everything to force feasibility
        open_set = list(range(m))
        assignment = tools['greedy_serve_nearest'](open_set)
        sol = tools['to_solution'](open_set, assignment)
        
    return sol