# MACE evolved heuristic 09/10 for problem: common_due_date_scheduling
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style hybrid heuristic.
    
    The problem structure for CDD scheduling is highly sensitive to n and 
    the penalty coefficients. Small instances (n <= 25) are effectively 
    solved via ILP or exhaustive search, while larger instances require 
    robust local optimization. 
    
    Regime 1 (Small): Use ILP or aggressive V-shape + local search.
    Regime 2 (Large): Use SA-based global search (B) to avoid entrapment 
                      in local minima typical of large combinatorial spaces.
    """
    start_time = time.time()
    n = tools['n_jobs']()
    
    # Heuristic Dispatch:
    # ILP is highly performant for n <= 25.
    # For n > 25, the problem space is too large for ILP; use SA (B-style)
    # to maintain exploration, as A-style deterministic search often gets 
    # stuck in large-instance basins.
    
    if n <= 25:
        # Attempt ILP for optimality, fallback to V-shape + deterministic search
        ilp_sol = tools['ilp_cdd'](time_limit_s=min(time_limit_s * 0.5, 10.0))
        if ilp_sol:
            return {'schedule': ilp_sol}
        
        # A-style: Deterministic local search refinement
        best_schedule = tools['v_shape_construct']()
        best_penalty = tools['compute_total_penalty'](best_schedule)
        
        refined = tools['apply_insertion_search'](best_schedule, time_limit_s=time_limit_s * 0.4)
        return {'schedule': refined}
    
    else:
        # B-style: Simulated Annealing (SA) for large combinatorial spaces
        best_schedule = tools['v_shape_construct']()
        best_penalty = tools['compute_total_penalty'](best_schedule)
        
        current_schedule = list(best_schedule)
        current_penalty = best_penalty
        
        temp = 100.0
        cooling_rate = 0.99995
        
        while time.time() - start_time < time_limit_s * 0.85:
            if random.random() < 0.5:
                i, j = random.sample(range(n), 2)
                new_schedule = list(current_schedule)
                new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
            else:
                i, j = random.sample(range(n), 2)
                new_schedule = list(current_schedule)
                job = new_schedule.pop(i)
                new_schedule.insert(j, job)
                
            new_penalty = tools['compute_total_penalty'](new_schedule)
            delta = new_penalty - current_penalty
            
            if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
                current_schedule = new_schedule
                current_penalty = new_penalty
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_schedule = list(current_schedule)
            
            temp *= cooling_rate
            if temp < 0.01: temp = 50.0
        
        # Final polishing
        remaining = time_limit_s - (time.time() - start_time) - 0.05
        if remaining > 0:
            refined = tools['apply_insertion_search'](best_schedule, time_limit_s=remaining)
            return {'schedule': refined}
            
        return {'schedule': best_schedule}