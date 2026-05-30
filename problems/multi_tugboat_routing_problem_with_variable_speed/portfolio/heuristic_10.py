# MACE evolved heuristic 10/10 for problem: multi_tugboat_routing_problem_with_variable_speed
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved MTRSP-VS solver using a randomized greedy construction with
    a focus on task density and iterative refinement.
    """
    start_time = time.time()
    num_tasks = instance['num_tasks']
    num_tugboats = instance['num_tugboats']
    
    def get_empty_solution():
        return {
            'routes': {k: [] for k in range(num_tugboats)},
            'service_speeds': {},
            'start_times': {},
            'transit_speeds': {k: [] for k in range(num_tugboats)}
        }

    # Pre-calculate priority: tasks with tighter, earlier windows are harder to fit
    task_indices = list(range(num_tasks))
    task_indices.sort(key=lambda s: (instance['task_time_window_lower'][s], 
                                     instance['task_service_distance'][s]))

    best_solution = get_empty_solution()
    best_objective = float('inf')

    # Iterative construction
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.85:
        iteration += 1
        current_solution = get_empty_solution()
        
        # Stochastic greedy: bias towards tasks that are easier to fit
        # or have been prioritized by earlier windows.
        shuffled_tasks = list(task_indices)
        if iteration > 1:
            # Add some randomness to the order
            for i in range(len(shuffled_tasks)):
                if random.random() < 0.2:
                    swap_idx = random.randint(0, len(shuffled_tasks) - 1)
                    shuffled_tasks[i], shuffled_tasks[swap_idx] = shuffled_tasks[swap_idx], shuffled_tasks[i]

        for t_id in shuffled_tasks:
            # Try to find a slot for the task
            # Check solo assignments first
            best_assignment = None
            # Trial: try all speeds, prioritizing lower fuel (speed 0)
            for speed in [0, 1, 2]:
                assignment = tools['find_feasible_assignment'](t_id, current_solution, prefer_speed=speed)
                if assignment:
                    best_assignment = assignment
                    break
            
            if best_assignment:
                current_solution = tools['append_task_to_tug'](
                    current_solution, t_id, best_assignment['tug_id'],
                    best_assignment['start_time'], best_assignment['service_speed'],
                    best_assignment['transit_speed_to'], best_assignment['transit_speed_from']
                )
                current_solution['service_speeds'][t_id] = best_assignment['service_speed']
                current_solution['start_times'][t_id] = best_assignment['start_time']

        # Evaluate objective
        try:
            current_obj = tools['objective'](current_solution)
            if current_obj < best_objective:
                best_objective = current_obj
                best_solution = current_solution
        except:
            continue

    # Local improvement: Attempt to drop the most expensive tasks (if any)
    # or swap speeds to reduce fuel.
    if best_solution['service_speeds']:
        executed = list(best_solution['service_speeds'].keys())
        for _ in range(50):
            if time.time() - start_time > time_limit_s * 0.98:
                break
            
            # Simple perturbation: try to reduce speed of a random task
            t_id = random.choice(executed)
            if best_solution['service_speeds'][t_id] > 0:
                candidate = {
                    'routes': {k: list(v) for k, v in best_solution['routes'].items()},
                    'service_speeds': dict(best_solution['service_speeds']),
                    'start_times': dict(best_solution['start_times']),
                    'transit_speeds': {k: list(v) for k, v in best_solution['transit_speeds'].items()}
                }
                candidate['service_speeds'][t_id] -= 1
                feasible, _ = tools['is_feasible'](candidate)
                if feasible:
                    new_obj = tools['objective'](candidate)
                    if new_obj < best_objective:
                        best_solution = candidate
                        best_objective = new_obj
    
    return best_solution