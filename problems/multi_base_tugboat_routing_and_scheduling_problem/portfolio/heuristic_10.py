# MACE evolved heuristic 10/10 for problem: multi_base_tugboat_routing_and_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style heuristic:
    - Analyzes instance 'density' (tasks per tugboat) and 'flexibility' 
      (avg time window width).
    - If density is low and time windows are tight, uses a high-exploration 
      LNS (B-style) to handle resource contention.
    - If density is high or time windows are loose, uses a refined GRASP 
      (A-style) to greedily pack tasks efficiently.
    """
    start_time = time.time()
    n = instance['num_tasks']
    m = instance['num_tugboats']
    
    # Analyze instance features
    avg_window = sum(instance['task_time_window_upper'][i] - instance['task_time_window_lower'][i] 
                     for i in range(n)) / n
    density = n / max(1, m)
    
    # Regime selection
    # Tight windows + low density -> High contention, needs LNS (B)
    # Wide windows + high density -> Packing problem, needs GRASP (A)
    use_lns = (avg_window < 5.0) or (density < 2.0)

    def get_empty():
        return {'routes': [[] for _ in range(m)], 'task_tugboats': {}, 'task_start_times': {}}

    if use_lns:
        # B-style LNS
        def construct(tasks_to_insert, base_sol=None):
            sol = base_sol if base_sol else get_empty()
            tasks_sorted = sorted(tasks_to_insert, key=lambda t: instance['task_time_window_lower'][t-1])
            for t in tasks_sorted:
                insertion = tools['find_feasible_insertion'](t, sol, try_starts=8)
                if insertion:
                    sol = tools['append_task_to_route'](sol, t, insertion['tug_ids'], insertion['tau'])
            return sol

        all_tasks = list(range(1, n + 1))
        current_sol = construct(all_tasks)
        best_sol, best_obj = current_sol, tools['objective'](current_sol)
        
        while time.time() - start_time < time_limit_s * 0.92:
            served = list(current_sol['task_tugboats'].keys())
            if not served: current_sol = construct(all_tasks); continue
            
            num_remove = min(len(served), max(1, int(len(served) * 0.2)))
            to_remove = random.sample(served, num_remove)
            
            working = {
                'routes': [r[:] for r in current_sol['routes']],
                'task_tugboats': {k: v for k, v in current_sol['task_tugboats'].items() if k not in to_remove},
                'task_start_times': {k: v for k, v in current_sol['task_start_times'].items() if k not in to_remove}
            }
            for r in working['routes']: r[:] = [t for t in r if t not in to_remove]
            
            unserved = [t for t in all_tasks if t not in working['task_tugboats']]
            random.shuffle(unserved)
            working = construct(unserved, working)
            
            new_obj = tools['objective'](working)
            if new_obj < best_obj:
                best_obj, best_sol = new_obj, working
            if new_obj < tools['objective'](current_sol) or random.random() < 0.05:
                current_sol = working
        return best_sol

    else:
        # A-style GRASP (refined)
        tasks = list(range(1, n + 1))
        best_sol, best_obj = get_empty(), float('inf')
        
        while time.time() - start_time < time_limit_s * 0.85:
            sol = get_empty()
            random.shuffle(tasks)
            for t in tasks:
                ins = tools['find_feasible_insertion'](t, sol, try_starts=4)
                if ins:
                    new_sol = tools['append_task_to_route'](sol, t, ins['tug_ids'], ins['tau'])
                    if tools['objective'](new_sol) < tools['objective'](sol): sol = new_sol
            
            curr_obj = tools['objective'](sol)
            if curr_obj < best_obj:
                best_obj, best_sol = curr_obj, sol
        return best_sol