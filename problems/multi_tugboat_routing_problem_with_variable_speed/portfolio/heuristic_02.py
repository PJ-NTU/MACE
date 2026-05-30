# MACE evolved heuristic 02/10 for problem: multi_tugboat_routing_problem_with_variable_speed
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved heuristic: Randomized Multi-Start with Earliest Due Date (EDD) 
    priority and greedy local search.
    """
    start_time = time.time()
    n = instance['num_tasks']
    m = instance['num_tugboats']
    
    # Pre-calculate task priorities based on flexibility and urgency
    # Low upper window + tight window range = high priority
    task_priorities = []
    for s in range(n):
        urgency = instance['task_time_window_upper'][s]
        duration = instance['task_service_distance'][s] / 10.0 # medium speed
        score = urgency + duration
        task_priorities.append((score, s))
    task_priorities.sort()
    sorted_tasks = [s for _, s in task_priorities]

    def get_empty_sol():
        return {
            'routes': {k: [] for k in range(m)},
            'service_speeds': {},
            'start_times': {},
            'transit_speeds': {k: [] for k in range(m)}
        }

    best_sol = get_empty_sol()
    best_obj = tools['objective'](best_sol)

    # Iterative improvement loop
    while time.time() - start_time < time_limit_s * 0.9:
        # Randomized restart strategy
        current_sol = get_empty_sol()
        
        # Shuffle order with a bias towards high-priority (tight/early) tasks
        # We use a small amount of jitter on the sorted list
        shuffled_tasks = sorted_tasks[:]
        for i in range(len(shuffled_tasks)):
            if random.random() < 0.2:
                idx = random.randint(0, len(shuffled_tasks) - 1)
                shuffled_tasks[i], shuffled_tasks[idx] = shuffled_tasks[idx], shuffled_tasks[i]
        
        for s in shuffled_tasks:
            if time.time() - start_time > time_limit_s * 0.95:
                break
            
            # Try speeds 0, 1, 2. Lower speeds are often more fuel-efficient.
            assignment = None
            for speed in [0, 1, 2]:
                res = tools['find_feasible_assignment'](s, current_sol, prefer_speed=speed)
                if res:
                    assignment = res
                    break
            
            if assignment:
                new_sol = tools['append_task_to_tug'](current_sol, s, **assignment)
                if tools['is_feasible'](new_sol)[0]:
                    current_sol = new_sol
        
        # Evaluate performance
        current_obj = tools['objective'](current_sol)
        if current_obj < best_obj:
            best_obj = current_obj
            best_sol = current_sol
            
        if n == 0: break

    # Final check: try to add any skipped tasks to the best solution found
    remaining = [s for s in range(n) if s not in best_sol['service_speeds']]
    random.shuffle(remaining)
    for s in remaining:
        if time.time() - start_time > time_limit_s * 0.98:
            break
        for speed in [0, 1, 2]:
            res = tools['find_feasible_assignment'](s, best_sol, prefer_speed=speed)
            if res:
                new_sol = tools['append_task_to_tug'](best_sol, s, **res)
                if tools['is_feasible'](new_sol)[0]:
                    best_sol = new_sol
                    best_obj = tools['objective'](best_sol)
                    break

    return best_sol