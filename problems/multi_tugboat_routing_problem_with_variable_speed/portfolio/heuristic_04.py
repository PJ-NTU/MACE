# MACE evolved heuristic 04/10 for problem: multi_tugboat_routing_problem_with_variable_speed
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher heuristic for MTRSP-VS. 
    
    Logic:
    - If task density (tasks/tugboats) is low, we have more flexibility; 
      use A-style (randomized multi-start construction + greedy refinement) 
      to explore the search space.
    - If task density is high, the problem is tightly constrained;
      use B-style (time-window-sorted construction) to minimize 
      the risk of early-on congestion leading to infeasibility.
    """
    start_time = time.time()
    n = instance['num_tasks']
    m = instance['num_tugboats']
    
    # Feature: Task density
    density = n / max(m, 1)
    
    def get_empty_sol():
        return {
            'routes': {k: [] for k in range(m)},
            'service_speeds': {},
            'start_times': {},
            'transit_speeds': {k: [] for k in range(m)}
        }

    best_sol = get_empty_sol()
    best_obj = float('inf')

    # Dispatch strategy
    if density < 3.0:
        # Strategy A: Exploration-focused
        while time.time() - start_time < time_limit_s * 0.8:
            current_sol = get_empty_sol()
            tasks = list(range(n))
            random.shuffle(tasks)
            
            for s in tasks:
                if time.time() - start_time > time_limit_s * 0.9: break
                for speed in [0, 1]:
                    res = tools['find_feasible_assignment'](s, current_sol, prefer_speed=speed)
                    if res:
                        new_sol = tools['append_task_to_tug'](current_sol, s, **res)
                        if tools['is_feasible'](new_sol)[0]:
                            current_sol = new_sol
                            break
            
            obj = tools['objective'](current_sol)
            if obj < best_obj:
                best_obj = obj
                best_sol = current_sol
    else:
        # Strategy B: Constraint-aware (Time-Window Sorted)
        while time.time() - start_time < time_limit_s * 0.8:
            current_sol = get_empty_sol()
            # Sort by time window to prevent early-start conflicts
            tasks = sorted(range(n), key=lambda x: instance['task_time_window_lower'][x] + random.uniform(0, 2))
            
            for s in tasks:
                if time.time() - start_time > time_limit_s * 0.9: break
                pref = random.choice([0, 1])
                res = tools['find_feasible_assignment'](s, current_sol, prefer_speed=pref)
                if res:
                    new_sol = tools['append_task_to_tug'](current_sol, s, **res)
                    if tools['is_feasible'](new_sol)[0]:
                        current_sol = new_sol
            
            try:
                obj = tools['objective'](current_sol)
                if obj < best_obj:
                    best_obj = obj
                    best_sol = current_sol
            except:
                continue

    # Final pass: attempt to fill gaps
    unexecuted = [s for s in range(n) if s not in best_sol['service_speeds']]
    random.shuffle(unexecuted)
    for s in unexecuted:
        if time.time() - start_time > time_limit_s * 0.98: break
        res = tools['find_feasible_assignment'](s, best_sol, prefer_speed=0)
        if res:
            new_sol = tools['append_task_to_tug'](best_sol, s, **res)
            if tools['is_feasible'](new_sol)[0]:
                best_sol = new_sol
                
    return best_sol