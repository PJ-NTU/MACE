# MACE evolved heuristic 02/10 for problem: multi_tugboat_routing_and_scheduling_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A hybridized MTRSP heuristic using:
    1. A 'Difficulty-Aware' construction phase (from h_a).
    2. A 'Regret-based' destruction/repair mechanism (from h_b).
    3. An 'Adaptive Perturbation' loop with simulated annealing-like acceptance.
    """
    start_time = time.time()
    num_tasks = instance['num_tasks']
    num_tugboats = instance['num_tugboats']
    
    # Calculate task difficulty for construction order
    tasks = []
    for s in range(1, num_tasks + 1):
        a_s = instance['task_time_window_lower'][s - 1]
        b_s = instance['task_time_window_upper'][s - 1]
        hp = instance['task_min_horsepower'][s - 1]
        # Difficulty: narrow windows and high HP requirements are harder to satisfy
        difficulty = (b_s - a_s) + (hp / 100.0)
        tasks.append({'id': s, 'difficulty': difficulty})
    
    sorted_tasks = [t['id'] for t in sorted(tasks, key=lambda x: x['difficulty'])]
    
    def get_empty_sol():
        return {
            'routes': [[] for _ in range(num_tugboats)],
            'task_tugboats': {s: [] for s in range(1, num_tasks + 1)},
            'task_start_times': {s: 0.0 for s in range(1, num_tasks + 1)}
        }

    def build_sol(order):
        sol = get_empty_sol()
        for s in order:
            assignment = tools['find_feasible_assignment'](s, sol)
            if assignment:
                sol = tools['apply_task_assignment'](sol, s, assignment['tug_ids'], assignment['start_time'])
        return sol

    # Initialize
    current_sol = build_sol(sorted_tasks)
    best_sol = current_sol
    best_obj = tools['objective'](best_sol)
    
    stagnation = 0
    temp = 1000.0  # For simulated annealing acceptance

    while time.time() - start_time < time_limit_s * 0.95:
        # Destroy phase: Regret-based removal
        executed = [s for s in range(1, num_tasks + 1) if current_sol['task_tugboats'][s]]
        if not executed:
            current_sol = build_sol(sorted_tasks)
            continue
            
        # Destroy 15-35% of tasks
        removal_count = max(1, int(len(executed) * (0.15 + 0.2 * (stagnation / 20.0))))
        to_remove = random.sample(executed, min(len(executed), removal_count))
        
        # Build new candidate by removing tasks
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
        
        # Repair phase: Randomized greedy insertion
        unassigned = [s for s in range(1, num_tasks + 1) if not candidate['task_tugboats'][s]]
        random.shuffle(unassigned)
        for s in unassigned:
            assignment = tools['find_feasible_assignment'](s, candidate)
            if assignment:
                candidate = tools['apply_task_assignment'](candidate, s, assignment['tug_ids'], assignment['start_time'])
        
        # Acceptance logic
        new_obj = tools['objective'](candidate)
        delta = new_obj - tools['objective'](current_sol)
        
        if delta < 0 or random.random() < (2.718 ** (-delta / max(1.0, temp))):
            current_sol = candidate
            if new_obj < best_obj:
                best_obj = new_obj
                best_sol = candidate
                stagnation = max(0, stagnation - 2)
            else:
                stagnation += 1
        else:
            stagnation += 1
            
        temp *= 0.995 # Cool down
        if stagnation > 40:
            temp = 1000.0
            stagnation = 0
            
    return best_sol