# MACE evolved heuristic 01/10 for problem: multi_tugboat_routing_and_scheduling_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An improved MTRSP heuristic using Randomized Greedy Construction with 
    Iterated Local Search (ILS), featuring an 'Adaptive Perturbation' 
    mechanism that scales the removal count based on the current objective 
    improvement rate to better escape local optima.
    """
    start_time = time.time()
    num_tasks = instance['num_tasks']
    num_tugboats = instance['num_tugboats']
    
    # Pre-calculate task difficulty metrics
    tasks = []
    for s in range(1, num_tasks + 1):
        a_s = instance['task_time_window_lower'][s - 1]
        b_s = instance['task_time_window_upper'][s - 1]
        hp = instance['task_min_horsepower'][s - 1]
        difficulty = (b_s - a_s) + (hp / 100.0)
        tasks.append({'id': s, 'difficulty': difficulty})
    
    tasks_sorted = [t['id'] for t in sorted(tasks, key=lambda x: x['difficulty'])]
    
    def build_sol(order):
        sol = {
            'routes': [[] for _ in range(num_tugboats)],
            'task_tugboats': {s: [] for s in range(1, num_tasks + 1)},
            'task_start_times': {s: 0.0 for s in range(1, num_tasks + 1)}
        }
        for s in order:
            assignment = tools['find_feasible_assignment'](s, sol)
            if assignment:
                sol = tools['apply_task_assignment'](sol, s, assignment['tug_ids'], assignment['start_time'])
        return sol

    best_sol = build_sol(tasks_sorted)
    best_obj = tools['objective'](best_sol)
    
    # ILS Loop
    last_improvement_time = start_time
    stagnation_counter = 0
    
    while time.time() - start_time < time_limit_s * 0.95:
        # Adaptive Perturbation: Scale removal rate based on stagnation
        executed = [s for s in range(1, num_tasks + 1) if best_sol['task_tugboats'][s]]
        if not executed:
            break
            
        # Increase intensity of random removal if we have been stuck
        removal_ratio = 0.1 + (0.3 * min(1.0, stagnation_counter / 50.0))
        num_to_remove = max(1, int(len(executed) * removal_ratio))
        to_remove = random.sample(executed, num_to_remove)
        
        sol = {
            'routes': [list(r) for r in best_sol['routes']],
            'task_tugboats': {s: list(ts) for s, ts in best_sol['task_tugboats'].items()},
            'task_start_times': {s: ts for s, ts in best_sol['task_start_times'].items()}
        }
        
        for s in to_remove:
            sol['task_tugboats'][s] = []
            for k in range(num_tugboats):
                if s in sol['routes'][k]:
                    sol['routes'][k].remove(s)
        
        # Repair: Re-insert using a randomized priority order
        unassigned = [s for s in range(1, num_tasks + 1) if not sol['task_tugboats'][s]]
        random.shuffle(unassigned)
        
        for s in unassigned:
            assignment = tools['find_feasible_assignment'](s, sol)
            if assignment:
                sol = tools['apply_task_assignment'](sol, s, assignment['tug_ids'], assignment['start_time'])
        
        curr_obj = tools['objective'](sol)
        if curr_obj < best_obj:
            best_obj = curr_obj
            best_sol = sol
            stagnation_counter = max(0, stagnation_counter - 5)
        else:
            stagnation_counter += 1
            
    return best_sol