# MACE evolved heuristic 05/10 for problem: container_loading_with_weight_restrictions
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Constraint-Satisfaction Tabu Search' approach.
    
    Unlike the portfolio's shelf-packing or greedy-constructive approaches,
    this heuristic treats the problem as a combinatorial optimization task
    over a fixed-size 'active set' of box placements. It maintains a
    population of valid placements and uses a tabu-list to avoid
    re-visiting specific box-type/orientation configurations, while
    performing 'move/swap' operations on the x,y,z coordinates of existing 
    placements to optimize for volume without relying on shelf-based construction.
    """
    start_time = time.time()
    container = tools['container_dims']()
    box_types = instance['box_types']
    
    # 1. Warm start using the provided robust solver
    best_sol = tools['solve_default'](time_limit_s=time_limit_s * 0.2)
    best_util = best_sol['util']
    placements = list(best_sol['placements'])
    
    # Tabu list stores (box_type, orientation, x_bucket, y_bucket, z_bucket)
    # to prevent oscillating between similar spatial configurations
    tabu_list = {}
    
    def get_bucket(val, max_val):
        return int((val / max(1, max_val)) * 5)

    # 2. Iterative Local Search: Coordinate Perturbation
    while time.time() - start_time < time_limit_s * 0.9:
        if not placements:
            break
            
        # Pick a random placement to perturb
        idx = random.randrange(len(placements))
        old_p = placements[idx].copy()
        
        # Perturbation: shift coordinate within container bounds
        dx, dy, dz = tools['box_dims'](old_p['box_type'], old_p['orientation'])
        old_p['x'] = random.uniform(0, max(0, container[0] - dx))
        old_p['y'] = random.uniform(0, max(0, container[1] - dy))
        old_p['z'] = random.uniform(0, max(0, container[2] - dz))
        
        # Check Tabu
        key = (old_p['box_type'], old_p['orientation'], 
               get_bucket(old_p['x'], container[0]), 
               get_bucket(old_p['y'], container[1]), 
               get_bucket(old_p['z'], container[2]))
        
        if tabu_list.get(key, 0) > time.time():
            continue
            
        # Test feasibility
        new_placements = placements[:]
        new_placements[idx] = old_p
        
        sol = tools['make_solution'](new_placements)
        is_ok, _ = tools['is_feasible'](sol)
        
        if is_ok:
            util = tools['utilization'](new_placements)
            if util > best_util:
                best_util = util
                placements = new_placements
                # Update tabu list
                tabu_list[key] = time.time() + 0.1
        else:
            # Randomly remove an offending box to recover feasibility
            if len(placements) > 1:
                placements.pop(random.randrange(len(placements)))
                
    return tools['make_solution'](placements)