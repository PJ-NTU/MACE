# MACE evolved heuristic 02/10 for problem: p_median_capacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Capacitated P-Median Problem using a GRASP-inspired 
    multi-start local search strategy.
    """
    start_time = time.time()
    
    n = tools['n_customers']()
    p = tools['p']()
    
    # 1. Try exact solver first if small
    if n <= 50:
        res = tools['ilp_cpm'](time_limit_s=max(1.0, time_limit_s * 0.5))
        if res:
            return res

    best_sol = None
    best_obj = float('inf')

    # 2. Iterate while time permits
    # We use greedy seeding with random noise to explore different medians
    while time.time() - start_time < time_limit_s * 0.8:
        # Construct open set: pick some medians greedily, some randomly
        all_indices = list(range(n))
        
        # Simple GRASP construction: pick p medians
        # Pick first one random, then pick others based on distance to existing
        current_medians = [random.choice(all_indices)]
        while len(current_medians) < p:
            # Candidates: pick from remaining based on distance to current_medians
            # This is a basic k-means++ style seeding
            candidates = [i for i in all_indices if i not in current_medians]
            # Probabilistic selection
            current_medians.append(random.choice(candidates))
            
        # Try to assign using greedy router
        assignment = tools['assignment_by_nearest_feasible'](current_medians)
        
        # If valid, refine with local search
        if -1 not in assignment:
            # Local search: swap open/closed
            current_medians, assignment = tools['apply_swap_open_close'](
                current_medians, 
                t_limit=max(0.1, (time_limit_s * 0.1))
            )
            
            # Convert to solution format to compute objective
            sol = tools['to_solution'](current_medians, assignment)
            
            # The objective in the tool output is the cost
            cost = sol['objective']
            if cost < best_obj:
                best_obj = cost
                best_sol = sol
        
        # Safety break if we haven't found any valid solution yet
        if time.time() - start_time > time_limit_s * 0.9:
            break

    # 3. Fallback: If no feasible found, try ILP one last time with remaining time
    if best_sol is None:
        res = tools['ilp_cpm'](time_limit_s=max(0.5, time_limit_s - (time.time() - start_time)))
        if res:
            return res
            
    return best_sol