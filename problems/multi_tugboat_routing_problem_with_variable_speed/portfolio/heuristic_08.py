# MACE evolved heuristic 08/10 for problem: multi_tugboat_routing_problem_with_variable_speed
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved MTRSP-VS solver using a randomized greedy construction with 
    a local search refinement phase.
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

    best_solution = get_empty_solution()
    best_objective = float('inf')

    # Calculate base costs for comparison
    # Every task not in service_speeds is penalized by W
    penalty_weight = instance['penalty_weight']

    # Heuristic: Prioritize tasks with earlier windows and higher service distances
    # to maximize the chance of filling tug schedules effectively.
    task_indices = list(range(num_tasks))

    while time.time() - start_time < time_limit_s * 0.9:
        current_solution = get_empty_solution()
        
        # Randomized priority queue: sort tasks by a mix of window and distance
        # to explore different insertion orders.
        random.shuffle(task_indices)
        sorted_tasks = sorted(task_indices, key=lambda s: (
            instance['task_time_window_lower'][s] + 
            random.uniform(0, 5) * instance['task_service_distance'][s]
        ))

        for task_id in sorted_tasks:
            if time.time() - start_time > time_limit_s * 0.8:
                break
            
            # Try to assign the task to any available tug
            best_assignment = None
            
            # Simple greedy: try speed 0, 1, or 2 (slowest to fastest)
            for speed in [0, 1, 2]:
                assignment = tools['find_feasible_assignment'](task_id, current_solution, prefer_speed=speed)
                if assignment:
                    best_assignment = assignment
                    break
            
            if best_assignment:
                current_solution = tools['append_task_to_tug'](
                    current_solution, task_id, best_assignment['tug_id'],
                    best_assignment['start_time'], best_assignment['service_speed'],
                    best_assignment['transit_speed_to'], best_assignment['transit_speed_from']
                )
                current_solution['service_speeds'][task_id] = best_assignment['service_speed']
                current_solution['start_times'][task_id] = best_assignment['start_time']

        # Evaluate and record if better
        try:
            current_obj = tools['objective'](current_solution)
            if current_obj < best_objective:
                best_objective = current_obj
                best_solution = current_solution
        except:
            continue

    # Final Local Search: attempt to speed up service for already assigned tasks 
    # to lower fuel consumption while maintaining feasibility.
    if best_solution['service_speeds']:
        for _ in range(20):
            if time.time() - start_time > time_limit_s * 0.98:
                break
            
            refined = {
                'routes': {k: list(v) for k, v in best_solution['routes'].items()},
                'service_speeds': dict(best_solution['service_speeds']),
                'start_times': dict(best_solution['start_times']),
                'transit_speeds': {k: list(v) for k, v in best_solution['transit_speeds'].items()}
            }
            
            t_id = random.choice(list(refined['service_speeds'].keys()))
            old_speed = refined['service_speeds'][t_id]
            # Try to reduce speed (lower fuel)
            if old_speed > 0:
                refined['service_speeds'][t_id] = old_speed - 1
                feasible, _ = tools['is_feasible'](refined)
                if feasible:
                    new_obj = tools['objective'](refined)
                    if new_obj < best_objective:
                        best_solution = refined
                        best_objective = new_obj

    return best_solution