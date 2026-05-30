# MACE evolved heuristic 10/10 for problem: common_due_date_scheduling
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A hybrid heuristic for the Restricted Single-Machine Common Due Date Scheduling Problem.
    
    Modification: Enhanced the initial search phase by starting with a multi-start 
    approach comparing V-shape and WSPT constructions to pick the best starting point, 
    and using insertion-heavy local search to better navigate the search space.
    """
    start_time = time.time()
    n = tools['n_jobs']()
    
    # 1. Improved Initial State Selection
    s1 = tools['v_shape_construct']()
    s2 = tools['wspt_construct']()
    p1 = tools['compute_total_penalty'](s1)
    p2 = tools['compute_total_penalty'](s2)
    
    if p1 < p2:
        best_schedule, best_penalty = s1, p1
    else:
        best_schedule, best_penalty = s2, p2
    
    current_schedule = list(best_schedule)
    current_penalty = best_penalty
    
    # 2. Search Parameters
    temp = 100.0
    cooling_rate = 0.9999
    
    while time.time() - start_time < time_limit_s * 0.90:
        # Bias towards insertion (Or-opt) as it is generally more effective 
        # for this specific scheduling structure than simple swaps.
        if random.random() < 0.3:
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
    remaining = time_limit_s - (time.time() - start_time) - 0.05
    if remaining > 0:
        refined = tools['apply_insertion_search'](best_schedule, time_limit_s=remaining)
        pen = tools['compute_total_penalty'](refined)
        if pen < best_penalty:
            best_schedule = refined
            
    return {'schedule': best_schedule}