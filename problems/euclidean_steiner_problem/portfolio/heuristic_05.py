# MACE evolved heuristic 05/10 for problem: euclidean_steiner_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A hybrid heuristic that combines Fermat-point seeding for high-quality
    local structure with a Simulated Annealing wrapper for global search.
    
    Modified: Augmented the Simulated Annealing 'Add' action to prefer
    placing Steiner points at the centroids of dense clusters (via 3-terminal
    MST triples) rather than purely random Gaussian noise.
    """
    start_time = time.time()
    
    # 1. Initialization
    initial_steiner = tools['add_fermat_points_for_mst_triples'](min_improvement=1e-9)
    current_steiner = list(initial_steiner)
    current_len = tools['mst_length'](current_steiner)
    
    best_steiner = list(current_steiner)
    best_len = current_len
    
    points = instance.get("points", [])
    if not points:
        return {"steiner_points": []}
        
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    diag = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
    
    # 2. Simulated Annealing Phase
    temp = 0.01 * diag 
    cooling = 0.999
    
    while time.time() - start_time < time_limit_s * 0.90:
        action = random.random()
        new_steiner = list(current_steiner)
        
        if action < 0.3 and len(new_steiner) < len(points) * 2:
            # Weighted Mutation: Instead of pure random, sample an MST edge 
            # and place a Steiner point near the midpoint to bridge connectivity
            edges = tools['mst_edges'](new_steiner)
            if edges:
                u_idx, v_idx = random.choice(edges)
                all_pts = points + new_steiner
                p1, p2 = all_pts[u_idx], all_pts[v_idx]
                mid = ((p1[0] + p2[0]) / 2 + random.gauss(0, diag*0.05), 
                       (p1[1] + p2[1]) / 2 + random.gauss(0, diag*0.05))
                new_steiner.append(mid)
        
        elif action < 0.6 and len(new_steiner) > 0:
            new_steiner.pop(random.randrange(len(new_steiner)))
            
        elif len(new_steiner) > 0:
            idx = random.randrange(len(new_steiner))
            px, py = new_steiner[idx]
            new_steiner[idx] = (px + random.gauss(0, temp), py + random.gauss(0, temp))
        
        new_len = tools['mst_length'](new_steiner)
        
        delta = new_len - current_len
        if delta < 0 or (temp > 1e-7 and math.exp(-delta / (temp + 1e-9)) > random.random()):
            current_steiner = new_steiner
            current_len = new_len
            if current_len < best_len:
                best_len = current_len
                best_steiner = list(current_steiner)
        
        temp *= cooling
        
    # 3. Final Polish
    final_refined = tools['local_relocate_steiner'](
        best_steiner, 
        time_limit_s=max(0.1, time_limit_s - (time.time() - start_time)),
        step=0.01
    )
    
    return tools['make_solution'](final_refined)