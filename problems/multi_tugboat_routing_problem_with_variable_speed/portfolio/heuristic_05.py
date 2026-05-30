# MACE evolved heuristic 05/10 for problem: multi_tugboat_routing_problem_with_variable_speed
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved MTRSP-VS solver implementing a multi-start randomized greedy construction
    with a hill-climbing local search phase for speed optimization and task insertion.
    Modified: Replaced the static top-3 choice with a probability distribution based on 
    priority to increase diversity in the construction phase.
    """
    start_time = time.time()
    num_tasks = instance['num_tasks']
    num_tugboats = instance['num_tugboats']
    
    # Priority: Tasks with tighter windows are harder to fit, so prioritize them.
    def get_task_priority(s):
        slack = instance['task_time_window_upper'][s] - instance['task_time_window_lower'][s]
        return (instance['task_time_window_lower'][s], slack)

    tasks_sorted = sorted(range(num_tasks), key=get_task_priority)

    def create_empty():
        return {
            'routes': {k: [] for k in range(num_tugboats)},
            'service_speeds': {},
            'start_times': {},
            'transit_speeds': {k: [] for k in range(num_tugboats)}
        }

    best_solution = create_empty()
    best_obj = float('inf')

    # Multi-start loop
    while time.time() - start_time < time_limit_s * 0.7:
        current_sol = create_empty()
        
        # Randomized construction: use a weighted distribution to pick tasks
        remaining = list(tasks_sorted)
        while remaining:
            # Weighted random selection: bias towards top tasks but allow wider exploration
            # Use geometric distribution for selection: index 0 is most likely
            weights = [0.6**i for i in range(len(remaining))]
            idx = random.choices(range(len(remaining)), weights=weights, k=1)[0]
            s = remaining.pop(idx)
            
            # Attempt insertion
            assignment = tools['find_feasible_assignment'](s, current_sol, prefer_speed=0)
            if assignment:
                new_sol = tools['append_task_to_tug'](
                    current_sol, s, assignment['tug_id'],
                    assignment['start_time'], assignment['service_speed'],
                    assignment['transit_speed_to'], assignment['transit_speed_from']
                )
                if tools['is_feasible'](new_sol)[0]:
                    current_sol = new_sol
        
        try:
            curr_obj = tools['objective'](current_sol)
            if curr_obj < best_obj:
                best_obj = curr_obj
                best_solution = current_sol
        except:
            continue

    # Local Search refinement
    executed = set(best_solution['service_speeds'].keys())
    dropped = [s for s in range(num_tasks) if s not in executed]
    
    while time.time() - start_time < time_limit_s * 0.95:
        # Try inserting dropped tasks
        if dropped:
            s = random.choice(dropped)
            assignment = tools['find_feasible_assignment'](s, best_solution, prefer_speed=0)
            if assignment:
                new_sol = tools['append_task_to_tug'](
                    best_solution, s, assignment['tug_id'],
                    assignment['start_time'], assignment['service_speed'],
                    assignment['transit_speed_to'], assignment['transit_speed_from']
                )
                if tools['is_feasible'](new_sol)[0]:
                    best_solution = new_sol
                    dropped.remove(s)
                    best_obj = tools['objective'](best_solution)
                    continue

        # Try reducing speeds on random tasks
        if executed:
            s = random.choice(list(executed))
            if best_solution['service_speeds'][s] > 0:
                candidate = dict(best_solution)
                candidate['service_speeds'] = dict(best_solution['service_speeds'])
                candidate['service_speeds'][s] -= 1
                if tools['is_feasible'](candidate)[0]:
                    new_obj = tools['objective'](candidate)
                    if new_obj < best_obj:
                        best_solution = candidate
                        best_obj = new_obj
        
        if not dropped and random.random() < 0.05: 
            break
            
    return best_solution