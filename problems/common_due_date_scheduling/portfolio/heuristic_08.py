# MACE evolved heuristic 08/10 for problem: common_due_date_scheduling
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher heuristic for the Common Due Date Scheduling Problem.
    
    Hypothesis:
    - Small instances (n <= 30) are best handled by exact methods (ILP) or 
      aggressive deterministic local search.
    - Large instances (n > 30) benefit from the Simulated Annealing exploration 
      used in Parent B to avoid getting stuck in local optima of the V-shape.
    """
    start_time = time.time()
    n = tools['n_jobs']()
    
    # 1. Dispatch logic based on instance complexity
    if n <= 25:
        # Use ILP for small instances where it is guaranteed to find the optimum
        # if the time limit allows.
        res = tools['ilp_cdd'](time_limit_s=time_limit_s * 0.8)
        if res is not None:
            return {'schedule': res}
        # Fallback to deterministic local search if ILP times out
        schedule = tools['v_shape_construct']()
        schedule = tools['apply_insertion_search'](schedule, time_limit_s=time_limit_s * 0.1)
        return {'schedule': schedule}
    
    else:
        # For larger instances, use the stochastic exploration (Parent B style)
        best_schedule = tools['v_shape_construct']()
        best_penalty = tools['compute_total_penalty'](best_schedule)
        
        current_schedule = list(best_schedule)
        current_penalty = best_penalty
        
        temp = 100.0
        cooling_rate = 0.9999
        
        # Run SA until ~90% of time budget
        while time.time() - start_time < time_limit_s * 0.90:
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
            if temp < 0.01:
                temp = 50.0
        
        # Final polishing with deterministic insertion
        remaining = time_limit_s - (time.time() - start_time) - 0.05
        if remaining > 0:
            refined = tools['apply_insertion_search'](best_schedule, time_limit_s=remaining)
            if tools['compute_total_penalty'](refined) < best_penalty:
                best_schedule = refined
                
        return {'schedule': best_schedule}