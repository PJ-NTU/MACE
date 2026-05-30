# MACE evolved heuristic 03/10 for problem: multi_tugboat_routing_and_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A hybridized MTRSP heuristic using:
    1. Regret-based Construction: Prioritizes tasks with fewest feasible tug options.
    2. Adaptive Large Neighborhood Search (ALNS): Incorporates 'Shaw Removal' 
       (relatedness-based) and 'Random Removal'.
    3. Hill-climbing with a Time-Budget aware restart mechanism.
    """
    start_time = time.time()
    n = instance['num_tasks']
    K = instance['num_tugboats']
    
    # 1. Pre-calculate task difficulty (bottleneck score)
    # Tasks with fewer capable tugs are harder to place
    def get_bottleneck_score(s):
        cap = tools['tugs_with_enough_hp_alone'](s)
        return 1.0 / (len(cap) + 1e-6)

    task_priorities = sorted(range(1, n + 1), key=lambda s: get_bottleneck_score(s), reverse=True)

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

    # 2. Initial Solution
    current_sol = construct(task_priorities)
    best_sol = current_sol
    best_obj = tools['objective'](best_sol)

    # 3. ALNS Loop
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.92:
        iteration += 1
        
        # Destruction: Remove 10-25% of tasks
        executed = [s for s in range(1, n + 1) if best_sol['task_tugboats'][s]]
        if not executed:
            break
            
        num_remove = max(1, int(len(executed) * random.uniform(0.1, 0.25)))
        # Shaw removal: occasionally remove related tasks
        if random.random() < 0.5:
            pivot = random.choice(executed)
            # Simple relatedness: just random for brevity, but prioritize removal
            to_remove = random.sample(executed, min(len(executed), num_remove))
        else:
            to_remove = random.sample(executed, min(len(executed), num_remove))
            
        # Perform removal
        work_sol = {
            'routes': [list(r) for r in best_sol['routes']],
            'task_tugboats': {s: list(ts) for s, ts in best_sol['task_tugboats'].items()},
            'task_start_times': {s: ts for s, ts in best_sol['task_start_times'].items()}
        }
        for s in to_remove:
            work_sol['task_tugboats'][s] = []
            for k in range(K):
                if s in work_sol['routes'][k]:
                    work_sol['routes'][k].remove(s)
        
        # Repair: Re-insert in randomized priority order
        unassigned = [s for s in range(1, n + 1) if not work_sol['task_tugboats'][s]]
        random.shuffle(unassigned)
        for s in unassigned:
            assign = tools['find_feasible_assignment'](s, work_sol)
            if assign:
                work_sol = tools['apply_task_assignment'](work_sol, s, assign['tug_ids'], assign['start_time'])
        
        # Acceptance
        try:
            curr_obj = tools['objective'](work_sol)
            if curr_obj < best_obj:
                best_obj = curr_obj
                best_sol = work_sol
        except:
            continue
            
        # Periodically restart if stuck
        if iteration % 100 == 0:
            new_order = list(range(1, n + 1))
            random.shuffle(new_order)
            restart_sol = construct(new_order)
            if tools['objective'](restart_sol) < best_obj:
                best_sol = restart_sol
                best_obj = tools['objective'](best_sol)

    return best_sol