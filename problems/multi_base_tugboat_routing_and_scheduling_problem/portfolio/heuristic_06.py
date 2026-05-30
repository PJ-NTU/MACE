# MACE evolved heuristic 06/10 for problem: multi_base_tugboat_routing_and_scheduling_problem
import time
import random
import copy

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust heuristic for MTRSP-MB combining:
    1. Adaptive construction prioritizing constrained tasks.
    2. Iterative Local Search (ILS) with task-removal perturbation.
    3. Time-aware greedy optimization.
    """
    start_time = time.time()
    n = instance['num_tasks']
    m = instance['num_tugboats']
    
    # Sort tasks by difficulty: (window size) * (1 + service_time)
    # Smaller values are more constrained/harder to fit.
    task_difficulty = []
    for t in range(1, n + 1):
        window = instance['task_time_window_upper'][t - 1] - instance['task_time_window_lower'][t - 1]
        difficulty = window * (1.0 + instance['task_service_time'][t - 1])
        task_difficulty.append((difficulty, t))
    task_difficulty.sort()
    sorted_task_ids = [t for _, t in task_difficulty]

    def get_empty_sol():
        return {
            'routes': [[] for _ in range(m)],
            'task_tugboats': {},
            'task_start_times': {}
        }

    def construct(task_order):
        sol = get_empty_sol()
        for t in task_order:
            if time.time() - start_time > time_limit_s * 0.9:
                break
            insertion = tools['find_feasible_insertion'](t, sol, try_starts=8)
            if insertion:
                sol = tools['append_task_to_route'](
                    sol, t, insertion['tug_ids'], insertion['tau']
                )
        return sol

    best_sol = get_empty_sol()
    best_obj = float('inf')

    # Initial construction
    current_sol = construct(sorted_task_ids)
    if tools['is_feasible'](current_sol)[0]:
        best_sol = current_sol
        best_obj = tools['objective'](best_sol)

    # ILS Loop
    while time.time() - start_time < time_limit_s * 0.95:
        # Perturbation: Remove random tasks to escape local optima
        working_sol = {
            'routes': [r[:] for r in best_sol['routes']],
            'task_tugboats': best_sol['task_tugboats'].copy(),
            'task_start_times': best_sol['task_start_times'].copy()
        }
        
        served = list(working_sol['task_tugboats'].keys())
        if not served:
            working_sol = construct(sorted_task_ids)
        else:
            # Remove a random subset (1 to 4)
            num_remove = random.randint(1, min(4, len(served)))
            to_remove = random.sample(served, num_remove)
            for t in to_remove:
                del working_sol['task_tugboats'][t]
                del working_sol['task_start_times'][t]
                for r in working_sol['routes']:
                    if t in r:
                        r.remove(t)
            
            # Fill back with remaining tasks in random order
            unserved = [t for t in range(1, n+1) if t not in working_sol['task_tugboats']]
            random.shuffle(unserved)
            for t in unserved:
                if time.time() - start_time > time_limit_s * 0.95:
                    break
                insertion = tools['find_feasible_insertion'](t, working_sol, try_starts=5)
                if insertion:
                    working_sol = tools['append_task_to_route'](
                        working_sol, t, insertion['tug_ids'], insertion['tau']
                    )
        
        # Evaluate
        if tools['is_feasible'](working_sol)[0]:
            obj = tools['objective'](working_sol)
            if obj < best_obj:
                best_obj = obj
                best_sol = working_sol
        
        # Occasional full restart
        if random.random() < 0.1:
            shuffled = list(sorted_task_ids)
            random.shuffle(shuffled)
            current_sol = construct(shuffled)
            if tools['is_feasible'](current_sol)[0]:
                obj = tools['objective'](current_sol)
                if obj < best_obj:
                    best_obj = obj
                    best_sol = current_sol

    return best_sol