# MACE evolved heuristic 05/10 for problem: multi_tugboat_routing_and_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust MTRSP heuristic using a Regret-based Construction and 
    Iterative Local Search (ILS) with a focus on maximizing task coverage.
    """
    start_time = time.time()
    n = instance['num_tasks']
    K = instance['num_tugboats']
    
    # 1. Pre-calculate task metrics for smarter construction
    # We prioritize tasks that are 'harder' to schedule (bottleneck)
    # and tasks with earlier deadlines.
    def get_difficulty(s):
        idx = s - 1
        cap = len(tools['tugs_with_enough_hp_alone'](s))
        # Inverse of flexibility + tightness of time window
        return (1.0 / (cap + 1.0)) * (instance['task_time_window_upper'][idx])

    task_order = sorted(range(1, n + 1), key=lambda s: get_difficulty(s))

    def get_empty_sol():
        return {
            'routes': [[] for _ in range(K)],
            'task_tugboats': {s: [] for s in range(1, n + 1)},
            'task_start_times': {s: 0.0 for s in range(1, n + 1)}
        }

    def construct(order):
        sol = get_empty_sol()
        for s in order:
            assignment = tools['find_feasible_assignment'](s, sol)
            if assignment:
                sol = tools['apply_task_assignment'](sol, s, assignment['tug_ids'], assignment['start_time'])
        return sol

    # Initial best
    best_sol = construct(task_order)
    best_obj = tools['objective'](best_sol)

    # Hill-climbing / ILS loop
    while time.time() - start_time < time_limit_s * 0.95:
        # Perturbation: Remove a random subset of tasks (10-30%)
        executed = [s for s in range(1, n + 1) if best_sol['task_tugboats'][s]]
        if not executed:
            # Try a different random order if we have no tasks
            best_sol = construct(random.sample(range(1, n + 1), n))
            best_obj = tools['objective'](best_sol)
            continue
            
        num_remove = max(1, int(len(executed) * random.uniform(0.1, 0.3)))
        to_remove = random.sample(executed, num_remove)
        
        # Create work solution
        work_sol = {
            'routes': [list(r) for r in best_sol['routes']],
            'task_tugboats': {s: list(ts) for s, ts in best_sol['task_tugboats'].items()},
            'task_start_times': {s: ts for s, ts in best_sol['task_start_times'].items()}
        }
        for s in to_remove:
            work_sol['task_tugboats'][s] = []
            for k in range(K):
                if s in work_sol['routes'][k]:
                    work_sol['routes'][k].remove(s)
        
        # Local Repair: Re-insert in a randomized order to explore new assignments
        unassigned = [s for s in range(1, n + 1) if not work_sol['task_tugboats'][s]]
        random.shuffle(unassigned)
        for s in unassigned:
            assignment = tools['find_feasible_assignment'](s, work_sol)
            if assignment:
                work_sol = tools['apply_task_assignment'](work_sol, s, assignment['tug_ids'], assignment['start_time'])
        
        # Evaluate
        try:
            curr_obj = tools['objective'](work_sol)
            if curr_obj < best_obj:
                best_obj = curr_obj
                best_sol = work_sol
        except:
            continue

        # Occasional restart with different priority
        if random.random() < 0.05:
            shuffled_order = list(range(1, n + 1))
            random.shuffle(shuffled_order)
            restart_sol = construct(shuffled_order)
            if tools['objective'](restart_sol) < best_obj:
                best_sol = restart_sol
                best_obj = tools['objective'](best_sol)

    return best_sol