# MACE evolved heuristic 05/10 for problem: multi_base_tugboat_routing_and_scheduling_problem
import time
import random
import copy

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust MTRSP-MB solver utilizing a Tabu-inspired Local Search over 
    a randomized greedy base. It emphasizes high-quality construction 
    with adaptive perturbation.
    """
    start_time = time.time()
    n = instance['num_tasks']
    m = instance['num_tugboats']
    penalty_weight = instance['penalty_weight']
    
    # Pre-calculate task difficulty (window size) for informed heuristic
    task_ids = list(range(1, n + 1))
    def get_window(tid):
        return instance['task_time_window_upper'][tid - 1] - instance['task_time_window_lower'][tid - 1]
    
    # Sort by window tightness to prioritize harder tasks
    task_ids.sort(key=get_window)
    
    best_sol = {
        'routes': [[] for _ in range(m)],
        'task_tugboats': {},
        'task_start_times': {}
    }
    best_obj = penalty_weight * n
    
    def construct(randomness=0.15):
        sol = {
            'routes': [[] for _ in range(m)],
            'task_tugboats': {},
            'task_start_times': {}
        }
        order = list(task_ids)
        # Jitter the order to allow exploration
        for i in range(len(order)):
            if random.random() < randomness:
                idx = random.randint(0, len(order) - 1)
                order[i], order[idx] = order[idx], order[i]
        
        for t in order:
            if time.time() - start_time > time_limit_s * 0.90:
                break
            insertion = tools['find_feasible_insertion'](t, sol, try_starts=6)
            if insertion:
                sol = tools['append_task_to_route'](sol, t, insertion['tug_ids'], insertion['tau'])
        return sol

    # Initial construction
    current_best = construct()
    current_obj = tools['objective'](current_best)
    if current_obj < best_obj:
        best_sol, best_obj = current_best, current_obj

    # Iterative Improvement: Shake and Local Search
    # Perturb the best solution by removing tasks and re-attempting insertion
    while time.time() - start_time < time_limit_s * 0.95:
        # Generate candidate by removing a small subset
        candidate = {
            'routes': [r[:] for r in best_sol['routes']],
            'task_tugboats': best_sol['task_tugboats'].copy(),
            'task_start_times': best_sol['task_start_times'].copy()
        }
        
        served = list(candidate['task_tugboats'].keys())
        if not served:
            candidate = construct()
        else:
            # Remove 1-4 tasks for local search
            num_rem = random.randint(1, min(4, len(served)))
            to_remove = random.sample(served, num_rem)
            for t in to_remove:
                del candidate['task_tugboats'][t]
                del candidate['task_start_times'][t]
                for r in candidate['routes']:
                    if t in r:
                        r.remove(t)
            
            # Attempt to re-insert tasks, including previously unserved
            unserved = [t for t in range(1, n + 1) if t not in candidate['task_tugboats']]
            random.shuffle(unserved)
            for t in unserved:
                if time.time() - start_time > time_limit_s * 0.95:
                    break
                insertion = tools['find_feasible_insertion'](t, candidate, try_starts=6)
                if insertion:
                    candidate = tools['append_task_to_route'](
                        candidate, t, insertion['tug_ids'], insertion['tau']
                    )
        
        # Evaluate candidate
        is_feas, _ = tools['is_feasible'](candidate)
        if is_feas:
            cand_obj = tools['objective'](candidate)
            if cand_obj < best_obj:
                best_sol, best_obj = candidate, cand_obj
        
        # Periodic restart to avoid local optima
        if random.random() < 0.1:
            candidate = construct(randomness=0.3)
            is_feas, _ = tools['is_feasible'](candidate)
            if is_feas:
                cand_obj = tools['objective'](candidate)
                if cand_obj < best_obj:
                    best_sol, best_obj = candidate, cand_obj
                    
    return best_sol