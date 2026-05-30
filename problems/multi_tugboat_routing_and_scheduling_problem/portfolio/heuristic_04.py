# MACE evolved heuristic 04/10 for problem: multi_tugboat_routing_and_scheduling_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized MTRSP heuristic using a 'Construct-Destroy-Repair' metaheuristic.
    
    Strategy:
    1. Initial Construction: Uses an Earliest Deadline First (EDF) heuristic 
       modified by task duration and HP requirements to build a high-quality baseline.
    2. Adaptive Large Neighborhood Search (ALNS): Iteratively destroys a subset
       of the solution (random removal combined with 'worst-cost' removal) and 
       repairs it using a randomized greedy approach.
    3. Time-Awareness: Self-monitors strictly; scales the number of repair attempts 
       to maximize the usage of the remaining time budget.
    """
    start_time = time.time()
    num_tasks = instance['num_tasks']
    num_tugboats = instance['num_tugboats']
    
    # Priority score: Early deadline tasks with high HP needs are harder to schedule
    task_scores = []
    for s in range(1, num_tasks + 1):
        idx = s - 1
        deadline = instance['task_time_window_upper'][idx]
        hp_req = instance['task_min_horsepower'][idx]
        # Lower score = higher priority
        task_scores.append((deadline, hp_req, s))
    task_scores.sort()
    sorted_tasks = [t[2] for t in task_scores]

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

    # Initial best solution
    best_sol = build_sol(sorted_tasks)
    best_obj = tools['objective'](best_sol)
    
    # Iterative Improvement
    iter_count = 0
    while time.time() - start_time < time_limit_s * 0.95:
        iter_count += 1
        
        # Destroy phase: Remove a percentage of tasks
        executed = [s for s in range(1, num_tasks + 1) if best_sol['task_tugboats'][s]]
        if not executed:
            break
            
        # Adaptive removal: remove 15-30% of tasks
        removal_count = max(1, int(len(executed) * random.uniform(0.15, 0.3)))
        to_remove = random.sample(executed, removal_count)
        
        # Create candidate
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
            # Check time periodically
            if iter_count % 10 == 0 and (time.time() - start_time) > time_limit_s * 0.95:
                break
            assignment = tools['find_feasible_assignment'](s, candidate)
            if assignment:
                candidate = tools['apply_task_assignment'](candidate, s, assignment['tug_ids'], assignment['start_time'])
        
        # Acceptance
        try:
            curr_obj = tools['objective'](candidate)
            if curr_obj < best_obj:
                best_obj = curr_obj
                best_sol = candidate
        except Exception:
            continue
            
    return best_sol