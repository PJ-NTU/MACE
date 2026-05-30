# MACE evolved heuristic 06/10 for problem: euclidean_steiner_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A hybridized Steiner Tree solver that combines constructive Fermat point 
    seeding with a robust Simulated Annealing metaheuristic.
    """
    start_time = time.time()
    terminals = instance.get("points", [])
    if not terminals:
        return {"steiner_points": []}

    # 1. Warm-start with constructive Fermat points
    # This provides a high-quality baseline that local search would take too long to find.
    initial_steiner = tools['add_fermat_points_for_mst_triples'](min_improvement=1e-9)
    
    # 2. Setup state
    current_steiner = list(initial_steiner)
    current_len = tools['mst_length'](current_steiner)
    best_steiner = list(current_steiner)
    best_len = current_len
    
    # Bounding box for random perturbations
    min_x = min(p[0] for p in terminals)
    max_x = max(p[0] for p in terminals)
    min_y = min(p[1] for p in terminals)
    max_y = max(p[1] for p in terminals)
    bbox_size = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
    
    # Annealing parameters
    T = 0.05 * current_len  # Start temperature relative to current cost
    cooling_rate = 0.9997
    
    # 3. Simulated Annealing Loop
    # Hybridizes h_a's global exploration with h_b's structured constructive start.
    while time.time() - start_time < time_limit_s * 0.95:
        idx = random.randrange(len(current_steiner)) if current_steiner else -1
        
        # Define neighborhood: Perturb, Remove, or Add
        op = random.random()
        new_steiner = list(current_steiner)
        
        if op < 0.1 and len(new_steiner) < len(terminals):
            # Attempt to add a random point in the center area
            new_steiner.append((random.uniform(min_x, max_x), random.uniform(min_y, max_y)))
        elif op < 0.3 and idx != -1:
            # Remove a point
            new_steiner.pop(idx)
        elif idx != -1:
            # Perturb existing point
            p = new_steiner[idx]
            scale = bbox_size * 0.02 * (T / (0.05 * current_len + 1e-9) + 0.1)
            new_steiner[idx] = (p[0] + random.gauss(0, scale), p[1] + random.gauss(0, scale))
        else:
            continue

        new_len = tools['mst_length'](new_steiner)
        
        # Metropolis criterion
        delta = new_len - current_len
        if delta < 0 or (T > 1e-9 and math.exp(-delta / T) > random.random()):
            current_steiner = new_steiner
            current_len = new_len
            
            if current_len < best_len:
                best_len = current_len
                best_steiner = list(current_steiner)
        
        T *= cooling_rate
        
    # 4. Final Polish
    # Use the tools' built-in coordinate descent to refine the best set found
    polished = tools['local_relocate_steiner'](
        best_steiner, 
        time_limit_s=max(0.05, time_limit_s - (time.time() - start_time)),
        step=0.01
    )
    
    return tools['make_solution'](steiner_points=polished)