# MACE evolved heuristic 04/10 for problem: euclidean_steiner_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Modified Simulated Annealing: Instead of purely random point generation,
    we seed the initial Steiner points using the high-quality Fermat-point
    heuristic and bias the mutation toward the neighborhood of existing
    high-performing Steiner points.
    """
    start_time = time.time()
    points = instance.get("points", [])
    if not points:
        return {"steiner_points": []}

    # Initial state: Seed with Fermat points for better starting geometry
    current_steiner = tools['add_fermat_points_for_mst_triples']()
    current_len = tools['mst_length'](current_steiner)
    best_steiner = list(current_steiner)
    best_len = current_len

    # Bounding box for random perturbations
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    range_x, range_y = max_x - min_x, max_y - min_y

    def get_mutated_point(p):
        # Localized bias: perturb existing point instead of total random
        sigma_x = range_x * 0.1
        sigma_y = range_y * 0.1
        return (max(min_x, min(max_x, p[0] + random.gauss(0, sigma_x))),
                max(min_y, min(max_y, p[1] + random.gauss(0, sigma_y))))

    temp = 1.0
    cooling = 0.995
    
    while time.time() - start_time < time_limit_s * 0.95:
        # Move: 20% add, 20% remove, 60% relocate existing (local search focus)
        action = random.random()
        new_steiner = list(current_steiner)
        
        if action < 0.2 and len(new_steiner) < len(points) * 2:
            new_steiner.append((random.uniform(min_x, max_x), random.uniform(min_y, max_y)))
        elif action < 0.4 and len(new_steiner) > 0:
            new_steiner.pop(random.randrange(len(new_steiner)))
        elif len(new_steiner) > 0:
            idx = random.randrange(len(new_steiner))
            new_steiner[idx] = get_mutated_point(new_steiner[idx])
        else:
            continue

        new_len = tools['mst_length'](new_steiner)
        
        # Metropolis acceptance criterion
        delta = new_len - current_len
        if delta < 0 or (temp > 1e-6 and math.exp(-delta / (temp + 1e-9)) > random.random()):
            current_steiner = new_steiner
            current_len = new_len
            
            if current_len < best_len:
                best_len = current_len
                best_steiner = list(current_steiner)
        
        temp *= cooling

    # Final feasibility enforcement
    return tools['make_solution'](best_steiner)