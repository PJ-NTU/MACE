# MACE evolved heuristic 07/10 for problem: multi_tugboat_routing_and_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved MTRSP heuristic using a hybrid construction strategy and 
    a Multi-Start ALNS with Simulated Annealing acceptance.
    """
    start_time = time.time()
    num_tasks = instance['num_tasks']
    num_tugboats = instance['num_tugboats']
    
    def get_task_priority(s):
        # Prioritize tasks by "tightness" of their constraints
        a_s = instance['task_time_window_lower'][s - 1]
        b_s = instance['task_time_window_upper'][s - 1]
        hp = instance['task_min_horsepower'][s - 1]
        capable_tugs = [k for k in range(num_tugboats) if instance['tugboat_horsepower'][k] >= hp]
        # High priority for small windows and few capable tugs
        return (b_s - a_s) * 0.5 + (100.0 / (len(capable_tugs) + 1))

    # Pre-sort tasks by priority
    sorted_tasks = sorted(range(1, num_tasks + 1), key=get_task_priority)
    
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

    # Initial Best
    best_sol = build_sol(sorted_tasks)
    best_obj = tools['objective'](best_sol)
    
    current_sol = best_sol
    current_obj = best_obj
    
    # Annealing params
    temp = 1000.0
    cooling = 0.998
    
    while time.time() - start_time < time_limit_s * 0.95:
        executed = [s for s in range(1, num_tasks + 1) if current_sol['task_tugboats'][s]]
        if not executed:
            # Restart if empty
            current_sol = build_sol(sorted_tasks)
            continue
            
        # Destroy: Remove random subset weighted towards large tasks
        removal_size = random.randint(1, max(1, len(executed) // 3))
        to_remove = random.sample(executed, removal_size)
        
        candidate = {
            'routes': [list(r) for r in current_sol['routes']],
            'task_tugboats': {s: list(ts) for s, ts in current_sol['task_tugboats'].items()},
            'task_start_times': {s: ts for s, ts in current_sol['task_start_times'].items()}
        }
        
        for s in to_remove:
            candidate['task_tugboats'][s] = []
            for k in range(num_tugboats):
                if s in candidate['routes'][k]:
                    candidate['routes'][k].remove(s)
        
        # Repair: Randomized greedy
        unassigned = [s for s in range(1, num_tasks + 1) if not candidate['task_tugboats'][s]]
        random.shuffle(unassigned)
        for s in unassigned:
            assignment = tools['find_feasible_assignment'](s, candidate)
            if assignment:
                candidate = tools['apply_task_assignment'](candidate, s, assignment['tug_ids'], assignment['start_time'])
        
        # Acceptance
        try:
            cand_obj = tools['objective'](candidate)
            delta = cand_obj - current_obj
            
            if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / max(1.0, temp))):
                current_sol = candidate
                current_obj = cand_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_sol = candidate
            
            temp *= cooling
            if temp < 0.1: temp = 1000.0
        except:
            continue
            
    return best_sol