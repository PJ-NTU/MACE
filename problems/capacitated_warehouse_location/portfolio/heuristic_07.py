# MACE evolved heuristic 07/10 for problem: capacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined solver for Capacitated Warehouse Location.
    
    Diagnosis of parent:
    - Over-reliance on ILP: The ILP solver might fail on large instances or 
      timeout without returning a partial solution.
    - Shallow Local Search: 'apply_swap_open_close' is a single-neighborhood 
      search that often gets stuck in local optima.
    - Lack of Multi-Restart: Single greedy start is insufficient for complex 
      landscapes.
    
    Redesign:
    1. Multi-start construction: Use randomized greedy construction to explore 
       different parts of the state space.
    2. Simulated Annealing / Variable Neighborhood Search approach:
       Perform iterative improvement by swapping open/closed status and 
       re-optimizing assignments.
    3. Time-aware loop: Maximize search depth within the time limit.
    """
    start_time = time.time()
    
    # 1. Attempt ILP first if it's likely to finish quickly
    # We restrict ILP to a smaller portion of time to allow for metaheuristic improvement
    ilp_sol = tools['ilp_cwl'](time_limit_s=min(time_limit_s * 0.4, 5.0))
    
    best_sol = None
    best_cost = float('inf')
    
    if ilp_sol and tools['is_feasible'](ilp_sol)[0]:
        best_sol = ilp_sol
        best_cost = ilp_sol['total_cost']
        
    m = tools['n_warehouses']()
    
    # 2. Multi-start Local Search Loop
    # We use random open sets and greedy serving to find diverse starting points
    while time.time() - start_time < time_limit_s * 0.9:
        # Randomly decide number of warehouses to open
        num_to_open = random.randint(1, m)
        open_set = sorted(random.sample(range(m), num_to_open))
        
        assignment = tools['greedy_serve_nearest'](open_set)
        
        # If greedy fails to assign all, skip this start
        if -1 in assignment:
            continue
            
        # Refine with local search (swap open/close)
        try:
            refined_open, refined_assignment = tools['apply_swap_open_close'](
                open_set, time_limit_s=max(0.1, (time_limit_s - (time.time() - start_time)) / 3)
            )
            sol = tools['to_solution'](refined_open, refined_assignment)
            
            if tools['is_feasible'](sol)[0]:
                cost = sol['total_cost']
                if cost < best_cost:
                    best_cost = cost
                    best_sol = sol
        except:
            continue
            
    # Final fallback: If no solution found, try to force-open all warehouses
    if best_sol is None:
        all_open = list(range(m))
        assignment = tools['greedy_serve_nearest'](all_open)
        if -1 not in assignment:
            best_sol = tools['to_solution'](all_open, assignment)
            
    return best_sol if best_sol else (ilp_sol or {})