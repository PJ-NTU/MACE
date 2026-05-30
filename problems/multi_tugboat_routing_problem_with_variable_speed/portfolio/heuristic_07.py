# MACE evolved heuristic 07/10 for problem: multi_tugboat_routing_problem_with_variable_speed
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A dispatch-driven heuristic that selects between 'Exploration' (stochastic)
    and 'Constraint-Aware' (greedy, time-sorted) strategies based on the 
    tightness of time windows and task-to-tug capacity ratios.
    """
    start_time = time.time()
    n = instance['num_tasks']
    m = instance['num_tugboats']
    
    # Feature Engineering: Determine problem tightness
    # 1. Window tightness (avg window width / horizon)
    avg_width = sum(instance['task_time_window_upper'][i] - instance['task_time_window_lower'][i] 
                    for i in range(n)) / n
    horizon = instance['planning_horizon']
    tightness = avg_width / horizon
    
    # 2. Resource density (tasks per tug)
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

    # Dispatch Logic:
    # If tasks are scarce and windows are wide (density < 1.5, tightness > 0.3),
    # use exploration (A-style) to find optimal speed/routing combinations.
    # If tasks are packed or windows are tight (else), prioritize feasibility
    # via time-window sorting (B-style).
    use_exploration = (density < 1.5 and tightness > 0.3)
    
    while time.time() - start_time < time_limit_s * 0.9:
        current_sol = get_empty_sol()
        
        if use_exploration:
            tasks = list(range(n))
            random.shuffle(tasks)
        else:
            # Sort by earliest start time + small random jitter to allow exploration
            tasks = sorted(range(n), key=lambda x: instance['task_time_window_lower'][x] + random.uniform(0, 0.5 * horizon))
            
        for s in tasks:
            if time.time() - start_time > time_limit_s * 0.95: break
            
            # Try to assign task with a preference for lower speeds to conserve fuel
            found = False
            for speed in [0, 1]:
                res = tools['find_feasible_assignment'](s, current_sol, prefer_speed=speed)
                if res:
                    new_sol = tools['append_task_to_tug'](current_sol, s, **res)
                    if tools['is_feasible'](new_sol)[0]:
                        current_sol = new_sol
                        found = True
                        break
            
        try:
            obj = tools['objective'](current_sol)
            if obj < best_obj:
                best_obj = obj
                best_sol = current_sol
        except:
            continue

    # Final pass: attempt to fill remaining unexecuted tasks
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