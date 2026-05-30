# MACE evolved heuristic 09/10 for problem: multi_base_tugboat_routing_and_scheduling_problem
import time
import random
import copy

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved MTRSP-MB solver using a Randomized Greedy Construction 
    with Iterative Local Search (ILS) and a prioritized task insertion heuristic.
    """
    start_time = time.time()
    n = instance['num_tasks']
    m = instance['num_tugboats']
    
    # Initialize with the empty solution (all tasks unserved)
    best_sol = {
        'routes': [[] for _ in range(m)],
        'task_tugboats': {},
        'task_start_times': {}
    }
    best_obj = tools['objective'](best_sol)
    
    def get_unserved(sol):
        served = set(sol['task_tugboats'].keys())
        return [t for t in range(1, n + 1) if t not in served]

    # Construction phase: Randomized Greedy with task prioritization
    # Instead of random order, prioritize tasks with tighter time windows (earlier deadlines)
    def construct_greedy():
        current_sol = {
            'routes': [[] for _ in range(m)],
            'task_tugboats': {},
            'task_start_times': {}
        }
        # Prioritize tasks by upper time window (earliest deadline first)
        tasks = sorted(range(1, n + 1), key=lambda x: instance['task_time_window_upper'][x-1])
        # Add slight randomness to prioritization to explore different greedy paths
        tasks = sorted(tasks, key=lambda x: random.random() * instance['task_time_window_upper'][x-1])
        
        for t in tasks:
            if time.time() - start_time > time_limit_s * 0.4:
                break
            insertion = tools['find_feasible_insertion'](t, current_sol, try_starts=7)
            if insertion:
                current_sol = tools['append_task_to_route'](
                    current_sol, t, insertion['tug_ids'], insertion['tau']
                )
        return current_sol

    current_best = construct_greedy()
    current_best_obj = tools['objective'](current_best)
    
    if current_best_obj < best_obj:
        best_sol = current_best
        best_obj = current_best_obj

    while time.time() - start_time < time_limit_s * 0.95:
        # Perturbation: Remove a random subset of tasks
        working_sol = {
            'routes': [r[:] for r in best_sol['routes']],
            'task_tugboats': best_sol['task_tugboats'].copy(),
            'task_start_times': best_sol['task_start_times'].copy()
        }
        
        served_tasks = list(working_sol['task_tugboats'].keys())
        if not served_tasks:
            new_sol = construct_greedy()
        else:
            num_to_remove = random.randint(1, min(4, len(served_tasks)))
            to_remove = random.sample(served_tasks, num_to_remove)
            
            for t in to_remove:
                del working_sol['task_tugboats'][t]
                del working_sol['task_start_times'][t]
                for r in working_sol['routes']:
                    if t in r:
                        r.remove(t)
            
            unserved = get_unserved(working_sol)
            # Re-insert with prioritized random order
            unserved.sort(key=lambda x: instance['task_time_window_upper'][x-1] * random.random())
            for t in unserved:
                if time.time() - start_time > time_limit_s * 0.95:
                    break
                insertion = tools['find_feasible_insertion'](t, working_sol, try_starts=5)
                if insertion:
                    working_sol = tools['append_task_to_route'](
                        working_sol, t, insertion['tug_ids'], insertion['tau']
                    )
            new_sol = working_sol

        is_feas, _ = tools['is_feasible'](new_sol)
        if is_feas:
            new_obj = tools['objective'](new_sol)
            if new_obj < best_obj:
                best_obj = new_obj
                best_sol = new_sol
        
        if random.random() < 0.05:
            current_best = construct_greedy()
            
    return best_sol