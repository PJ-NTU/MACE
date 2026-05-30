# MACE evolved heuristic 01/10 for problem: multi_tugboat_routing_problem_with_variable_speed
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A randomized greedy insertion heuristic with adaptive task ordering
    and multi-start local search refinement.
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

    # Heuristic strategy: 
    # 1. Shuffle task order for randomized construction
    # 2. Prefer slower speeds initially to conserve fuel for more tasks
    # 3. Perform multiple construction passes within time budget
    
    tasks = list(range(num_tasks))
    
    while time.time() - start_time < time_limit_s * 0.85:
        current_solution = get_empty_solution()
        random.shuffle(tasks)
        
        for task_id in tasks:
            # Try to find a feasible assignment
            # Try speeds in order 0 (slow), 1 (medium), 2 (fast)
            assignment = None
            for speed in [0, 1, 2]:
                tmp = tools['find_feasible_assignment'](task_id, current_solution, prefer_speed=speed)
                if tmp:
                    assignment = tmp
                    break
            
            if assignment:
                new_sol = tools['append_task_to_tug'](
                    current_solution, task_id, assignment['tug_id'],
                    assignment['start_time'], assignment['service_speed'],
                    assignment['transit_speed_to'], assignment['transit_speed_from']
                )
                
                # Verify feasibility
                is_feas, _ = tools['is_feasible'](new_sol)
                if is_feas:
                    current_solution = new_sol
        
        # Evaluate objective
        current_obj = tools['objective'](current_solution)
        if current_obj < best_objective:
            best_objective = current_obj
            best_solution = current_solution
            
        # If we have a perfect score (all tasks done minimum fuel), break early
        if best_objective < (instance['penalty_weight'] * 0.1):
            break

    # Local Refinement: Attempt to swap task speeds to lower fuel costs
    # for the best found solution.
    if best_solution['service_speeds']:
        refined_solution = dict(best_solution)
        executed = list(best_solution['service_speeds'].keys())
        
        for _ in range(50):
            if time.time() - start_time > time_limit_s * 0.98:
                break
            t_id = random.choice(executed)
            old_speed = refined_solution['service_speeds'][t_id]
            
            # Try lower speed
            if old_speed > 0:
                refined_solution['service_speeds'][t_id] = old_speed - 1
                is_feas, _ = tools['is_feasible'](refined_solution)
                if is_feas and tools['objective'](refined_solution) < tools['objective'](best_solution):
                    best_solution = dict(refined_solution)
                    best_objective = tools['objective'](best_solution)
                else:
                    refined_solution['service_speeds'][t_id] = old_speed
                    
    return best_solution