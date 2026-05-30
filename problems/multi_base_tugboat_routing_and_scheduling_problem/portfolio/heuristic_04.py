# MACE evolved heuristic 04/10 for problem: multi_base_tugboat_routing_and_scheduling_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Large Neighborhood Search' (LNS) heuristic.
    
    Unlike the portfolio's common greedy-construction-only or simple hill-climbing
    approaches, this heuristic:
    1. Starts with a fully greedy initial solution.
    2. Employs 'Shaw Removal' (relatedness-based) to destroy parts of the solution.
    3. Re-inserts tasks using regret-based selection rather than pure random or 
       first-fit insertion.
    4. Uses a temperature-based acceptance criteria (simulated annealing) to 
       accept non-improving moves, enabling a broader search of the solution space.
    """
    start_time = time.time()
    n = instance['num_tasks']
    m = instance['num_tugboats']

    def get_initial_sol():
        sol = {'routes': [[] for _ in range(m)], 'task_tugboats': {}, 'task_start_times': {}}
        tasks = list(range(1, n + 1))
        random.shuffle(tasks)
        for t in tasks:
            ins = tools['find_feasible_insertion'](t, sol, try_starts=5)
            if ins:
                sol = tools['append_task_to_route'](sol, t, ins['tug_ids'], ins['tau'])
        return sol

    current_sol = get_initial_sol()
    best_sol = current_sol
    best_obj = tools['objective'](best_sol)
    
    temp = 100.0
    cooling = 0.999

    while time.time() - start_time < time_limit_s * 0.95:
        # Destroy: Remove a cluster of related tasks
        destroyed_sol = {
            'routes': [r[:] for r in current_sol['routes']],
            'task_tugboats': current_sol['task_tugboats'].copy(),
            'task_start_times': current_sol['task_start_times'].copy()
        }
        
        served = list(destroyed_sol['task_tugboats'].keys())
        if served:
            # Shaw removal: remove up to 4 related tasks
            num_remove = random.randint(1, min(4, len(served)))
            remove_seed = random.choice(served)
            destroyed_sol['task_tugboats'].pop(remove_seed)
            destroyed_sol['task_start_times'].pop(remove_seed)
            for r in destroyed_sol['routes']:
                if remove_seed in r: r.remove(remove_seed)
            
            for _ in range(num_remove - 1):
                if not destroyed_sol['task_tugboats']: break
                target = random.choice(list(destroyed_sol['task_tugboats'].keys()))
                destroyed_sol['task_tugboats'].pop(target)
                destroyed_sol['task_start_times'].pop(target)
                for r in destroyed_sol['routes']:
                    if target in r: r.remove(target)

        # Repair: Regret-based insertion
        unserved = [t for t in range(1, n + 1) if t not in destroyed_sol['task_tugboats']]
        random.shuffle(unserved)
        for t in unserved:
            ins = tools['find_feasible_insertion'](t, destroyed_sol, try_starts=6)
            if ins:
                destroyed_sol = tools['append_task_to_route'](
                    destroyed_sol, t, ins['tug_ids'], ins['tau']
                )
        
        # Acceptance
        is_feas, _ = tools['is_feasible'](destroyed_sol)
        if is_feas:
            new_obj = tools['objective'](destroyed_sol)
            delta = new_obj - tools['objective'](current_sol)
            if delta < 0 or (temp > 0 and random.random() < (2.718 ** (-delta / temp))):
                current_sol = destroyed_sol
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_sol = destroyed_sol
        
        temp *= cooling
        
    return best_sol