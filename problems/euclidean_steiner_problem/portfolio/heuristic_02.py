# MACE evolved heuristic 02/10 for problem: euclidean_steiner_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A hybrid heuristic that combines Fermat-point seeding for high-quality
    local structure with a Simulated Annealing wrapper for global search.
    
    The strategy:
    1. Start with a solid foundation by generating Fermat points from the MST.
    2. Use Simulated Annealing to explore the configuration space of these
       Steiner points.
    3. The state space includes adding, removing, and small-step jittering
       of Steiner points, allowing the algorithm to escape local optima
       found by pure coordinate descent.
    """
    start_time = time.time()
    
    # 1. Initialization
    # Start with a high-quality baseline using deterministic Fermat points
    initial_steiner = tools['add_fermat_points_for_mst_triples'](min_improvement=1e-9)
    current_steiner = list(initial_steiner)
    current_len = tools['mst_length'](current_steiner)
    
    best_steiner = list(current_steiner)
    best_len = current_len
    
    # Define bounding box for random perturbations
    points = instance.get("points", [])
    if not points:
        return {"steiner_points": []}
        
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    diag = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
    
    # 2. Simulated Annealing Phase
    temp = 0.01 * diag # Initial temperature scaled by problem size
    cooling = 0.999
    
    while time.time() - start_time < time_limit_s * 0.90:
        action = random.random()
        new_steiner = list(current_steiner)
        
        if action < 0.3 and len(new_steiner) < len(points) * 2:
            # Add point: focus on areas around existing terminals
            ref = points[random.randrange(len(points))]
            new_steiner.append((ref[0] + random.gauss(0, diag*0.1), ref[1] + random.gauss(0, diag*0.1)))
        
        elif action < 0.6 and len(new_steiner) > 0:
            # Remove point: clean up inefficient Steiner points
            new_steiner.pop(random.randrange(len(new_steiner)))
            
        elif len(new_steiner) > 0:
            # Perturb point: explore local neighborhood
            idx = random.randrange(len(new_steiner))
            px, py = new_steiner[idx]
            new_steiner[idx] = (px + random.gauss(0, temp), py + random.gauss(0, temp))
        
        new_len = tools['mst_length'](new_steiner)
        
        # Metropolis-Hastings acceptance
        delta = new_len - current_len
        if delta < 0 or (temp > 1e-7 and math.exp(-delta / (temp + 1e-9)) > random.random()):
            current_steiner = new_steiner
            current_len = new_len
            if current_len < best_len:
                best_len = current_len
                best_steiner = list(current_steiner)
        
        temp *= cooling
        
    # 3. Final Polish
    # Use coordinate descent to snap points into the exact local minimum
    final_refined = tools['local_relocate_steiner'](
        best_steiner, 
        time_limit_s=max(0.1, time_limit_s - (time.time() - start_time)),
        step=0.01
    )
    
    return tools['make_solution'](final_refined)