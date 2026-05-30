# MACE evolved heuristic 09/10 for problem: multi_tugboat_routing_problem_with_variable_speed
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style heuristic for MTRSP-VS.
    
    Dispatch Hypothesis:
    - Low-density instances (tasks/tugs < 2.5) favor randomized greedy insertion 
      (Strategy A) because the search space is sparse; rapid exploration of 
      permutations yields better results.
    - High-density instances (tasks/tugs >= 2.5) favor time-window-aware 
      priority queue construction (Strategy B) to mitigate bottlenecking 
      and constraint violations.
    """
    start_time = time.time()
    n = instance['num_tasks']
    m = instance['num_tugboats']
    
    def get_empty_sol():
        return {
            'routes': {k: [] for k in range(m)},
            'service_speeds': {},
            'start_times': {},
            'transit_speeds': {k: [] for k in range(m)}
        }

    def run_strategy_randomized_greedy():
        """Strategy A: Pure randomized greedy insertion with local speed tuning."""
        best_sol = get_empty_sol()
        best_obj = float('inf')
        while time.time() - start_time < time_limit_s * 0.7:
            current_sol = get_empty_sol()
            tasks = list(range(n))
            random.shuffle(tasks)
            for s in tasks:
                if time.time() - start_time > time_limit_s * 0.8: break
                for speed in [0, 1, 2]:
                    res = tools['find_feasible_assignment'](s, current_sol, prefer_speed=speed)
                    if res:
                        new_sol = tools['append_task_to_tug'](current_sol, s, **res)
                        if tools['is_feasible'](new_sol)[0]:
                            current_sol = new_sol
                            break
            try:
                obj = tools['objective'](current_sol)
                if obj < best_obj:
                    best_obj = obj
                    best_sol = current_sol
            except: continue
        return best_sol

    def run_strategy_window_priority():
        """Strategy B: Time-window-sorted construction with selection noise."""
        best_sol = get_empty_sol()
        best_obj = float('inf')
        tasks_sorted = sorted(range(n), key=lambda x: instance['task_time_window_lower'][x])
        while time.time() - start_time < time_limit_s * 0.7:
            current_sol = get_empty_sol()
            remaining = list(tasks_sorted)
            while remaining:
                # Use a decaying probability to select from sorted tasks
                weights = [0.8**i for i in range(len(remaining))]
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

    # 1. Dispatch
    density = n / max(m, 1)
    if density < 2.5:
        best_sol = run_strategy_randomized_greedy()
    else:
        best_sol = run_strategy_window_priority()

    # 2. Post-Optimization: Greedy filling of remaining unexecuted tasks
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