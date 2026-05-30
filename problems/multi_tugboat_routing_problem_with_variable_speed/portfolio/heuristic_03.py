# MACE evolved heuristic 03/10 for problem: multi_tugboat_routing_problem_with_variable_speed
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A dispatching heuristic for MTRSP-VS.
    
    Hypothesis:
    - If the number of tasks is small relative to tugboats, or if the time windows 
      are very tight (high density/constraint pressure), the problem behaves like 
      a packing problem. A-style local search refinement is more effective.
    - If the number of tasks is large, the problem behaves like a routing problem. 
      B-style randomized construction with window-priority sorting is more effective.
    """
    start_time = time.time()
    num_tasks = instance['num_tasks']
    num_tugboats = instance['num_tugboats']
    
    # Calculate constraint density: ratio of tasks to tugs and tightness of windows
    avg_window = sum(instance['task_time_window_upper'][i] - instance['task_time_window_lower'][i] 
                     for i in range(num_tasks)) / max(1, num_tasks)
    
    # Dispatch decision
    # If tasks/tugs ratio is low or windows are extremely tight, use refined approach (A)
    # Otherwise use the construction-heavy approach (B)
    use_refined_strategy = (num_tasks / max(1, num_tugboats) < 2.0) or (avg_window < 5.0)

    def get_empty_solution():
        return {
            'routes': {k: [] for k in range(num_tugboats)},
            'service_speeds': {},
            'start_times': {},
            'transit_speeds': {k: [] for k in range(num_tugboats)}
        }

    best_solution = get_empty_solution()
    best_objective = float('inf')

    while time.time() - start_time < time_limit_s * 0.85:
        current_solution = get_empty_solution()
        
        if not use_refined_strategy:
            # B-style: Priority on time windows
            tasks = sorted(range(num_tasks), key=lambda s: instance['task_time_window_lower'][s] + random.uniform(0, 2))
        else:
            # A-style: Pure randomization
            tasks = list(range(num_tasks))
            random.shuffle(tasks)
        
        for task_id in tasks:
            if time.time() - start_time > time_limit_s * 0.95:
                break
            
            # Try to find a feasible assignment
            # Vary speed preference to explore fuel vs time trade-offs
            pref = 0 if use_refined_strategy else random.choice([0, 1, 2])
            assignment = tools['find_feasible_assignment'](task_id, current_solution, prefer_speed=pref)
            
            if assignment:
                current_solution = tools['append_task_to_tug'](
                    current_solution, task_id, assignment['tug_id'],
                    assignment['start_time'], assignment['service_speed'],
                    assignment['transit_speed_to'], assignment['transit_speed_from']
                )
                current_solution['service_speeds'][task_id] = assignment['service_speed']
                current_solution['start_times'][task_id] = assignment['start_time']
        
        # Evaluate objective
        try:
            current_obj = tools['objective'](current_solution)
            if current_obj < best_objective:
                best_objective = current_obj
                best_solution = current_solution
        except:
            continue

    # Post-process refinement for A-style strategy
    if use_refined_strategy and best_solution['service_speeds']:
        refined_solution = dict(best_solution)
        executed = list(best_solution['service_speeds'].keys())
        for _ in range(30):
            if time.time() - start_time > time_limit_s * 0.98: break
            t_id = random.choice(executed)
            old_speed = refined_solution['service_speeds'][t_id]
            if old_speed > 0:
                refined_solution['service_speeds'][t_id] = old_speed - 1
                if tools['is_feasible'](refined_solution)[0]:
                    new_obj = tools['objective'](refined_solution)
                    if new_obj < best_objective:
                        best_solution = dict(refined_solution)
                        best_objective = new_obj
                    else:
                        refined_solution['service_speeds'][t_id] = old_speed
                else:
                    refined_solution['service_speeds'][t_id] = old_speed
                    
    return best_solution