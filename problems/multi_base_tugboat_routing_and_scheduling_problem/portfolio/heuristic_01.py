# MACE evolved heuristic 01/10 for problem: multi_base_tugboat_routing_and_scheduling_problem
import time
import random
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A multi-start randomized adaptive search procedure (GRASP) with 
    a focus on task-set optimization. Unlike the greedy portfolio, 
    this uses a 'build-down' approach: starts with a set of tasks and 
    attempts to prune the least efficient ones if they hinder the 
    overall fuel/penalty balance, or swaps them.
    """
    start_time = time.time()
    
    num_tasks = instance['num_tasks']
    num_tugboats = instance['num_tugboats']
    
    best_solution = {
        'routes': [[] for _ in range(num_tugboats)],
        'task_tugboats': {},
        'task_start_times': {}
    }
    best_obj = tools['objective'](best_solution)
    
    # Adaptive Selection: Weight tasks by their potential "value"
    # (Inverse of service time or penalty weight)
    tasks = list(range(1, num_tasks + 1))
    
    # Multi-start iteration
    while time.time() - start_time < time_limit_s * 0.85:
        # Randomized construction
        current_solution = {
            'routes': [[] for _ in range(num_tugboats)],
            'task_tugboats': {},
            'task_start_times': {}
        }
        
        # Shuffle tasks with bias: prioritize shorter service times to fit more
        random.shuffle(tasks)
        
        for task_id in tasks:
            if time.time() - start_time > time_limit_s * 0.9:
                break
            
            # Use the tool to find insertion, but with randomized try_starts
            insertion = tools['find_feasible_insertion'](task_id, current_solution, try_starts=3)
            
            if insertion:
                new_sol = tools['append_task_to_route'](
                    current_solution, 
                    task_id, 
                    insertion['tug_ids'], 
                    insertion['tau']
                )
                # Only accept if it improves objective (hill climbing)
                if tools['objective'](new_sol) < tools['objective'](current_solution):
                    current_solution = new_sol
        
        # Evaluate
        current_obj = tools['objective'](current_solution)
        if current_obj < best_obj:
            best_obj = current_obj
            best_solution = current_solution
            
        # Perturbation: Remove a random task to escape local optima
        if len(current_solution['task_tugboats']) > 0:
            tasks_served = list(current_solution['task_tugboats'].keys())
            task_to_remove = random.choice(tasks_served)
            
            # Reconstruct without the removed task
            new_routes = [[] for _ in range(num_tugboats)]
            new_tugboats = {}
            new_start_times = {}
            
            for s, tugs in current_solution['task_tugboats'].items():
                if s == task_to_remove: continue
                # Re-add using the same logic as construction
                insertion = tools['find_feasible_insertion'](s, {'routes': new_routes, 'task_tugboats': new_tugboats, 'task_start_times': new_start_times}, try_starts=2)
                if insertion:
                    # Apply
                    for k in insertion['tug_ids']: new_routes[k].append(s)
                    new_tugboats[s] = insertion['tug_ids']
                    new_start_times[s] = insertion['tau']
            
            current_solution = {
                'routes': new_routes,
                'task_tugboats': new_tugboats,
                'task_start_times': new_start_times
            }

    return best_solution