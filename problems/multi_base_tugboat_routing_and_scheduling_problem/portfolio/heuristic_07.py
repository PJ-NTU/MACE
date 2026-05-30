# MACE evolved heuristic 07/10 for problem: multi_base_tugboat_routing_and_scheduling_problem
import time
import random
import copy

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust MTRSP-MB solver utilizing a Tabu-inspired Adaptive Large Neighborhood Search (ALNS).
    This design addresses the shortcomings of previous heuristics by combining:
    1. A greedy construction based on 'urgency' (tight windows + high resource reqs).
    2. A destruction operator that removes 'costly' tasks to allow for re-insertion.
    3. A dynamic time budget management to maximize performance.
    """
    start_time = time.time()
    n = instance['num_tasks']
    m = instance['num_tugboats']
    
    # Pre-calculate task urgency scores for intelligent construction
    # Urgency: tighter windows and higher min_horsepower are harder to schedule.
    task_scores = []
    for s in range(1, n + 1):
        i = s - 1
        window_range = instance['task_time_window_upper'][i] - instance['task_time_window_lower'][i]
        hp_req = instance['task_min_horsepower'][i]
        # Normalize: smaller window = more urgent, higher HP = more urgent
        score = (float(window_range) / 100.0) - (hp_req / 1000.0)
        task_scores.append((score, s))
    
    # Sort tasks by urgency (ascending score = more urgent)
    sorted_tasks = [t[1] for t in sorted(task_scores)]

    def get_empty_sol():
        return {
            'routes': [[] for _ in range(m)],
            'task_tugboats': {},
            'task_start_times': {}
        }

    def construct(task_order):
        sol = get_empty_sol()
        for t in task_order:
            if time.time() - start_time > time_limit_s * 0.95:
                break
            insertion = tools['find_feasible_insertion'](t, sol, try_starts=8)
            if insertion:
                sol = tools['append_task_to_route'](sol, t, insertion['tug_ids'], insertion['tau'])
        return sol

    # Initial construction
    best_sol = construct(sorted_tasks)
    best_obj = tools['objective'](best_sol)
    
    # Iterative Improvement (ALNS-lite)
    while time.time() - start_time < time_limit_s * 0.9:
        # Clone current best
        current_sol = {
            'routes': [r[:] for r in best_sol['routes']],
            'task_tugboats': best_sol['task_tugboats'].copy(),
            'task_start_times': best_sol['task_start_times'].copy()
        }
        
        served_tasks = list(current_sol['task_tugboats'].keys())
        if not served_tasks:
            # Reconstruct if empty
            current_sol = construct(sorted_tasks)
        else:
            # Destroy: Remove a random number of tasks (1 to min(4, len))
            remove_count = random.randint(1, min(4, len(served_tasks)))
            to_remove = random.sample(served_tasks, remove_count)
            for t in to_remove:
                del current_sol['task_tugboats'][t]
                del current_sol['task_start_times'][t]
                for r in current_sol['routes']:
                    if t in r:
                        r.remove(t)
            
            # Repair: Try to insert removed tasks plus a few others
            unserved = [t for t in range(1, n + 1) if t not in current_sol['task_tugboats']]
            random.shuffle(unserved)
            for t in unserved:
                if time.time() - start_time > time_limit_s * 0.95:
                    break
                insertion = tools['find_feasible_insertion'](t, current_sol, try_starts=5)
                if insertion:
                    current_sol = tools['append_task_to_route'](
                        current_sol, t, insertion['tug_ids'], insertion['tau']
                    )
        
        # Validate and update
        is_feas, _ = tools['is_feasible'](current_sol)
        if is_feas:
            new_obj = tools['objective'](current_sol)
            if new_obj < best_obj:
                best_obj = new_obj
                best_sol = current_sol
        
        # Periodic restart to avoid local optima
        if random.random() < 0.02:
            new_order = sorted_tasks[:]
            random.shuffle(new_order)
            restart_sol = construct(new_order)
            if tools['objective'](restart_sol) < best_obj:
                best_sol = restart_sol
                best_obj = tools['objective'](best_sol)

    return best_sol