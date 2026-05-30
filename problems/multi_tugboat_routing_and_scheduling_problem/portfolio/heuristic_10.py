# MACE evolved heuristic 10/10 for problem: multi_tugboat_routing_and_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized MTRSP solver using:
    1. Regret-based Construction: Prioritizes tasks that are harder to place.
    2. Adaptive Large Neighborhood Search (ALNS): Incorporates Randomized Removal 
       and a strong Simulated Annealing acceptance criteria.
    3. Time-Budget aware execution: Dynamically adjusts search intensity.
    """
    start_time = time.time()
    n = instance['num_tasks']
    K = instance['num_tugboats']
    
    # 1. Pre-calculate task difficulty (bottleneck score)
    # Tasks with fewer capable tugs or tighter windows are harder to place
    def get_difficulty(s):
        _, _, T_s = tools['task_time_window'](s)
        a_s = instance['task_time_window_lower'][s - 1]
        b_s = instance['task_time_window_upper'][s - 1]
        cap = len(tools['tugs_with_enough_hp_alone'](s))
        return (b_s - a_s) / (T_s + 0.1) + (1.0 / (cap + 0.1))

    # Sort tasks by difficulty: hardest first
    task_priorities = sorted(range(1, n + 1), key=lambda s: get_difficulty(s))

    def get_empty_sol():
        return {
            'routes': [[] for _ in range(K)],
            'task_tugboats': {s: [] for s in range(1, n + 1)},
            'task_start_times': {s: 0.0 for s in range(1, n + 1)}
        }

    def construct(order):
        sol = get_empty_sol()
        for s in order:
            assign = tools['find_feasible_assignment'](s, sol)
            if assign:
                sol = tools['apply_task_assignment'](sol, s, assign['tug_ids'], assign['start_time'])
        return sol

    # Initialize
    best_sol = construct(task_priorities)
    best_obj = tools['objective'](best_sol)
    current_sol = best_sol
    current_obj = best_obj
    
    # Annealing params
    temp = 1.0
    cooling = 0.999
    
    # 2. Main Search Loop
    while time.time() - start_time < time_limit_s * 0.95:
        # Destruction: Remove random portion of executed tasks
        executed = [s for s in range(1, n + 1) if current_sol['task_tugboats'][s]]
        if not executed:
            current_sol = construct(task_priorities)
            continue
            
        num_remove = max(1, int(len(executed) * random.uniform(0.1, 0.3)))
        to_remove = random.sample(executed, num_remove)
        
        # Perform removal
        work_sol = {
            'routes': [list(r) for r in current_sol['routes']],
            'task_tugboats': {s: list(ts) for s, ts in current_sol['task_tugboats'].items()},
            'task_start_times': {s: ts for s, ts in current_sol['task_start_times'].items()}
        }
        for s in to_remove:
            work_sol['task_tugboats'][s] = []
            for k in range(K):
                if s in work_sol['routes'][k]:
                    work_sol['routes'][k].remove(s)
        
        # Repair: Randomized greedy insertion
        unassigned = [s for s in range(1, n + 1) if not work_sol['task_tugboats'][s]]
        random.shuffle(unassigned)
        for s in unassigned:
            assign = tools['find_feasible_assignment'](s, work_sol)
            if assign:
                work_sol = tools['apply_task_assignment'](work_sol, s, assign['tug_ids'], assign['start_time'])
        
        # Acceptance
        try:
            new_obj = tools['objective'](work_sol)
            delta = new_obj - current_obj
            
            if delta < 0 or (temp > 1e-6 and random.random() < math.exp(-delta / (temp * 1000))):
                current_sol = work_sol
                current_obj = new_obj
                if current_obj < best_obj:
                    best_sol = work_sol
                    best_obj = current_obj
            
            temp *= cooling
        except:
            continue
            
    return best_sol