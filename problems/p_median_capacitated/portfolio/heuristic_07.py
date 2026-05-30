# MACE evolved heuristic 07/10 for problem: p_median_capacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined Capacitated P-Median heuristic:
    1. Prioritize ILP for small/medium instances.
    2. Multi-start construction using a mix of greedy seeding and randomized diversity.
    3. Iterative Local Search (ILS) with aggressive time management and 
       a fallback mechanism to ensure feasibility.
    """
    start_time = time.time()
    n = tools['n_customers']()
    p = tools['p']()
    
    # 1. ILP phase: High quality for small/medium instances
    ilp_time = min(time_limit_s * 0.35, 6.0)
    best_sol = tools['ilp_cpm'](time_limit_s=ilp_time)
    best_obj = best_sol['objective'] if best_sol else float('inf')
    
    # 2. Iterative Improvement phase
    # Use greedy seeds as a starting base, then random diversification
    seeds = [tools['greedy_p_picks_by_distance']()]
    
    # Main loop
    while time.time() - start_time < time_limit_s * 0.95:
        # Select medians: alternate greedy and random
        if seeds:
            current_medians = seeds.pop(0)
        else:
            # Diversify by picking random subset of facilities
            current_medians = random.sample(range(n), p)
            
        # Try to construct a feasible assignment
        assignments = tools['assignment_by_nearest_feasible'](current_medians)
        
        # If construction fails, attempt a simple fix: replace a random median
        if -1 in assignments:
            # Replace one random median with one random unused facility
            unused = list(set(range(n)) - set(current_medians))
            if unused:
                current_medians[random.randrange(p)] = random.choice(unused)
                assignments = tools['assignment_by_nearest_feasible'](current_medians)
            else:
                continue

        if -1 in assignments:
            continue
            
        # Refine using Local Search
        remaining = time_limit_s - (time.time() - start_time)
        if remaining < 0.05:
            break
            
        # Perform swap-based local search
        ls_time = min(remaining * 0.6, 2.0)
        new_medians, new_assignments = tools['apply_swap_open_close'](
            current_medians, t_limit=ls_time
        )
        
        candidate = tools['to_solution'](new_medians, new_assignments)
        
        is_f, _ = tools['is_feasible'](candidate)
        if is_f:
            obj = candidate['objective']
            if obj < best_obj:
                best_obj = obj
                best_sol = candidate
                
    # 3. Final Fallback: If no solution found, force a greedy construction
    if best_sol is None:
        greedy_medians = tools['greedy_p_picks_by_distance']()
        greedy_assign = tools['assignment_by_nearest_feasible'](greedy_medians)
        if -1 not in greedy_assign:
            best_sol = tools['to_solution'](greedy_medians, greedy_assign)
            
    return best_sol if best_sol else {}