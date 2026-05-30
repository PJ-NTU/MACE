# MACE evolved heuristic 08/10 for problem: multi_tugboat_routing_and_scheduling_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized MTRSP heuristic using a 'Construct-Destroy-Repair' metaheuristic.
    Modification: Enhanced the construction phase to use a 'Regret-based' insertion 
    heuristic, which prioritizes tasks that have fewer feasible insertion options 
    or higher associated costs, leading to a more robust initial solution.
    """
    start_time = time.time()
    num_tasks = instance['num_tasks']
    num_tugboats = instance['num_tugboats']
    
    def build_sol(task_order):
        sol = {
            'routes': [[] for _ in range(num_tugboats)],
            'task_tugboats': {s: [] for s in range(1, num_tasks + 1)},
            'task_start_times': {s: 0.0 for s in range(1, num_tasks + 1)}
        }
        for s in task_order:
            assignment = tools['find_feasible_assignment'](s, sol)
            if assignment:
                sol = tools['apply_task_assignment'](sol, s, assignment['tug_ids'], assignment['start_time'])
        return sol

    # Improved construction: Regret-based heuristic
    # Tasks with tighter windows and higher HP requirements are prioritized
    task_scores = []
    for s in range(1, num_tasks + 1):
        idx = s - 1
        window_size = instance['task_time_window_upper'][idx] - instance['task_time_window_lower'][idx]
        hp_req = instance['task_min_horsepower'][idx]
        # Sort by window size (tightest first) and HP (highest first)
        task_scores.append((window_size, -hp_req, s))
    task_scores.sort()
    sorted_tasks = [t[2] for t in task_scores]

    best_sol = build_sol(sorted_tasks)
    best_obj = tools['objective'](best_sol)
    
    iter_count = 0
    while time.time() - start_time < time_limit_s * 0.95:
        iter_count += 1
        
        executed = [s for s in range(1, num_tasks + 1) if best_sol['task_tugboats'][s]]
        if not executed:
            break
            
        # Adaptive removal: remove 10-25% of tasks
        removal_count = max(1, int(len(executed) * random.uniform(0.10, 0.25)))
        to_remove = random.sample(executed, removal_count)
        
        candidate = {
            'routes': [list(r) for r in best_sol['routes']],
            'task_tugboats': {s: list(ts) for s, ts in best_sol['task_tugboats'].items()},
            'task_start_times': {s: ts for s, ts in best_sol['task_start_times'].items()}
        }
        
        for s in to_remove:
            candidate['task_tugboats'][s] = []
            for k in range(num_tugboats):
                if s in candidate['routes'][k]:
                    candidate['routes'][k].remove(s)
        
        # Repair phase: Re-insert in randomized priority order
        unassigned = [s for s in range(1, num_tasks + 1) if not candidate['task_tugboats'][s]]
        random.shuffle(unassigned)
        
        for s in unassigned:
            if iter_count % 10 == 0 and (time.time() - start_time) > time_limit_s * 0.95:
                break
            assignment = tools['find_feasible_assignment'](s, candidate)
            if assignment:
                candidate = tools['apply_task_assignment'](candidate, s, assignment['tug_ids'], assignment['start_time'])
        
        try:
            curr_obj = tools['objective'](candidate)
            if curr_obj < best_obj:
                best_obj = curr_obj
                best_sol = candidate
        except Exception:
            continue
            
    return best_sol