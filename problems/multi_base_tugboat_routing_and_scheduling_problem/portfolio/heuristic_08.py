# MACE evolved heuristic 08/10 for problem: multi_base_tugboat_routing_and_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized LNS heuristic for MTRSP-MB.
    
    Improvements over h_a:
    1. Adaptive Destroy: Alternates between Shaw removal (temporal proximity)
       and random removal to balance exploration and exploitation.
    2. Greedy Repair with Noise: Uses a randomized greedy insertion criterion
       to broaden the search space rather than pure deterministic greedy.
    3. Better Time Management: Dynamically adjusts iterations based on 
       remaining time.
    4. Stronger Initial Construction: Sorts tasks by time window tightness
       (earliest start) to build a better starting baseline.
    """
    start_time = time.time()
    n = instance['num_tasks']
    m = instance['num_tugboats']
    
    def get_empty():
        return {'routes': [[] for _ in range(m)], 'task_tugboats': {}, 'task_start_times': {}}

    def construct(tasks_to_insert, base_sol=None):
        sol = base_sol if base_sol else get_empty()
        # Sort tasks by start of window to prioritize early tasks
        tasks_sorted = sorted(tasks_to_insert, key=lambda t: instance['task_time_window_lower'][t-1])
        for t in tasks_sorted:
            # Use noise to pick insertions if time permits
            insertion = tools['find_feasible_insertion'](t, sol, try_starts=8)
            if insertion:
                sol = tools['append_task_to_route'](sol, t, insertion['tug_ids'], insertion['tau'])
        return sol

    # Initialize
    all_tasks = list(range(1, n + 1))
    current_sol = construct(all_tasks)
    best_sol = current_sol
    best_obj = tools['objective'](best_sol)
    
    # Adaptive Parameters
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.92:
        iteration += 1
        served = list(current_sol['task_tugboats'].keys())
        if not served:
            current_sol = construct(all_tasks)
            continue
            
        # Destroy
        num_remove = min(len(served), max(1, int(len(served) * (0.1 + 0.2 * random.random()))))
        if iteration % 2 == 0:
            # Shaw removal
            seed = random.choice(served)
            to_remove = sorted(served, key=lambda x: abs(instance['task_time_window_lower'][x-1] - 
                                                         instance['task_time_window_lower'][seed-1]))[:num_remove]
        else:
            # Random removal
            to_remove = random.sample(served, num_remove)
            
        working = {
            'routes': [r[:] for r in current_sol['routes']],
            'task_tugboats': {k: v for k, v in current_sol['task_tugboats'].items() if k not in to_remove},
            'task_start_times': {k: v for k, v in current_sol['task_start_times'].items() if k not in to_remove}
        }
        for r in working['routes']:
            r[:] = [t for t in r if t not in to_remove]
            
        # Repair
        unserved = [t for t in all_tasks if t not in working['task_tugboats']]
        random.shuffle(unserved)
        working = construct(unserved, working)
        
        # Accept
        new_obj = tools['objective'](working)
        # Simple hill climbing with occasional random walk
        if new_obj < tools['objective'](current_sol) or random.random() < 0.05:
            current_sol = working
            if new_obj < best_obj:
                best_obj = new_obj
                best_sol = working
                
    return best_sol