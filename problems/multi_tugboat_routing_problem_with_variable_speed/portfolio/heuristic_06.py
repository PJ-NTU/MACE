# MACE evolved heuristic 06/10 for problem: multi_tugboat_routing_problem_with_variable_speed
import time
import random
import math
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style heuristic for MTRSP-VS.
    
    Hypothesis:
    - High-density instances (n >> m) are highly sensitive to task ordering 
      (time-window conflicts). Strategy B's priority-weighted construction is robust here.
    - Low-density/Small-scale instances (n ≈ m or few tasks) benefit from 
      randomized exploration, as the search space is less constrained by 
      congestion and more by finding the 'best' assignment to minimize fuel.
    """
    start_time = time.time()
    n = instance['num_tasks']
    m = instance['num_tugboats']
    
    # Feature: Task density
    density = n / max(m, 1) if m > 0 else n
    
    def get_empty_sol():
        return {
            'routes': {k: [] for k in range(m)},
            'service_speeds': {},
            'start_times': {},
            'transit_speeds': {k: [] for k in range(m)}
        }

    def run_strategy_a():
        best_sol = get_empty_sol()
        best_obj = float('inf')
        while time.time() - start_time < time_limit_s * 0.7:
            current_sol = get_empty_sol()
            tasks = list(range(n))
            random.shuffle(tasks)
            for s in tasks:
                if time.time() - start_time > time_limit_s * 0.85: break
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
        return best_sol

    def run_strategy_b():
        best_sol = get_empty_sol()
        best_obj = float('inf')
        tasks_sorted = sorted(range(n), key=lambda x: instance['task_time_window_lower'][x])
        while time.time() - start_time < time_limit_s * 0.7:
            current_sol = get_empty_sol()
            remaining = list(tasks_sorted)
            while remaining:
                weights = [0.7**i for i in range(len(remaining))]
                idx = random.choices(range(len(remaining)), weights=weights, k=1)[0]
                s = remaining.pop(idx)
                res = tools['find_feasible_assignment'](s, current_sol, prefer_speed=0)
                if res:
                    new_sol = tools['append_task_to_tug'](current_sol, s, **res)
                    if tools['is_feasible'](new_sol)[0]:
                        current_sol = new_sol
            try:
                obj = tools['objective'](current_sol)
                if obj < best_obj:
                    best_obj = obj
                    best_sol = current_sol
            except: continue
        return best_sol

    # Dispatch based on density hypothesis
    if density < 2.5:
        best_sol = run_strategy_a()
    else:
        best_sol = run_strategy_b()

    # Final refinement: attempt to insert missed tasks
    executed = set(best_sol['service_speeds'].keys())
    dropped = [s for s in range(n) if s not in executed]
    random.shuffle(dropped)
    for s in dropped:
        if time.time() - start_time > time_limit_s * 0.95: break
        res = tools['find_feasible_assignment'](s, best_sol, prefer_speed=0)
        if res:
            new_sol = tools['append_task_to_tug'](best_sol, s, **res)
            if tools['is_feasible'](new_sol)[0]:
                best_sol = new_sol
                
    return best_sol