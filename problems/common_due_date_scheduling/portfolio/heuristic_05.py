# MACE evolved heuristic 05/10 for problem: common_due_date_scheduling
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A hybrid heuristic for the Restricted Single-Machine Common Due Date Scheduling Problem.
    
    Combines:
    1. V-shape construction for a high-quality initial solution.
    2. Iterative Local Search (ILS) with two complementary neighborhoods (swap/insertion).
    3. Simulated Annealing (SA) trajectory to escape local minima during the search.
    """
    start_time = time.time()
    n = tools['n_jobs']()
    
    # 1. Initial State
    # V-shape is optimal for the unconstrained structure; we start there.
    best_schedule = tools['v_shape_construct']()
    best_penalty = tools['compute_total_penalty'](best_schedule)
    
    current_schedule = list(best_schedule)
    current_penalty = best_penalty
    
    # 2. Search Parameters
    # SA parameters for exploration
    temp = 100.0
    cooling_rate = 0.9999
    
    while time.time() - start_time < time_limit_s * 0.90:
        # Choose neighborhood: 50% swap, 50% insertion
        if random.random() < 0.5:
            # Swap move
            i, j = random.sample(range(n), 2)
            new_schedule = list(current_schedule)
            new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
        else:
            # Insertion move (Or-opt)
            i, j = random.sample(range(n), 2)
            new_schedule = list(current_schedule)
            job = new_schedule.pop(i)
            new_schedule.insert(j, job)
            
        new_penalty = tools['compute_total_penalty'](new_schedule)
        
        # Metropolis acceptance criterion
        delta = new_penalty - current_penalty
        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
            current_schedule = new_schedule
            current_penalty = new_penalty
            
            # Global best tracker
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_schedule = list(current_schedule)
        
        # Cool down
        temp *= cooling_rate
        
        # Periodically re-heat to escape deep local minima
        if temp < 0.01:
            temp = 50.0
            
    # 3. Final Polishing
    # Exhaust remaining time with deterministic local search to ensure convergence
    remaining = time_limit_s - (time.time() - start_time) - 0.05
    if remaining > 0:
        refined = tools['apply_insertion_search'](best_schedule, time_limit_s=remaining)
        pen = tools['compute_total_penalty'](refined)
        if pen < best_penalty:
            best_schedule = refined
            
    return {'schedule': best_schedule}