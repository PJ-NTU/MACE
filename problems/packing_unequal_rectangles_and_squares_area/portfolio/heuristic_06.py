# MACE evolved heuristic 06/10 for problem: packing_unequal_rectangles_and_squares_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized heuristic using a two-stage strategy:
    1. Deterministic construction (Area-First Greedy) to secure a high baseline.
    2. Adaptive Large Neighborhood Search (ALNS) with a temperature-controlled
       rejection sampling logic to escape local optima.
    """
    start_time = time.time()
    n = instance['n']
    rotation = instance['rotation']
    
    # 1. Warm start: Area-first greedy is the strongest baseline
    best_placements = tools['greedy_by_area_first'](time_limit_s=time_limit_s * 0.15, grid=25)
    best_area = tools['total_area'](best_placements)
    
    current_placements = list(best_placements)
    current_area = best_area
    
    # Pre-sort indices by area descending to focus destructive moves on smaller items
    # or to prioritize larger items during reconstruction.
    items_by_area = sorted(range(n), key=lambda i: tools['item_area'](i), reverse=True)
    
    # Parameters for adaptive search
    temp = 1.0
    cooling_rate = 0.99995
    
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.95:
        iteration += 1
        
        # Select a neighborhood operation:
        # 0: Remove random subset and try to re-insert (Destruction/Construction)
        # 1: Single item swap/perturbation
        work_placements = list(current_placements)
        
        if random.random() < 0.4:
            # Destructive move: Remove random items
            num_to_remove = random.randint(1, max(1, n // 4))
            indices_to_remove = random.sample(range(n), num_to_remove)
            for idx in indices_to_remove:
                work_placements[idx] = (-1.0, -1.0, 0.0)
            
            # Reconstruct greedily
            for idx in items_by_area:
                if work_placements[idx][0] == -1:
                    thetas = [0, 90] if rotation else [0]
                    random.shuffle(thetas)
                    for theta in thetas:
                        pos = tools['random_feasible_position'](idx, theta=theta, placements=work_placements, max_tries=30)
                        if pos:
                            new_state = tools['try_place'](work_placements, idx, pos[0], pos[1], theta)
                            if new_state:
                                work_placements = new_state
                                break
        else:
            # Local move: Perturb single item
            idx = random.randint(0, n - 1)
            work_placements[idx] = (-1.0, -1.0, 0.0)
            theta = random.choice([0, 90]) if rotation else 0
            pos = tools['random_feasible_position'](idx, theta=theta, placements=work_placements, max_tries=50)
            if pos:
                new_state = tools['try_place'](work_placements, idx, pos[0], pos[1], theta)
                if new_state:
                    work_placements = new_state

        # Evaluation
        new_area = tools['total_area'](work_placements)
        
        # Metropolis acceptance criterion
        delta = new_area - current_area
        if delta > 0 or (temp > 1e-7 and math.exp(delta / (temp + 1e-9)) > random.random()):
            is_feasible, _ = tools['is_feasible']({'placements': work_placements})
            if is_feasible:
                current_placements = list(work_placements)
                current_area = new_area
                if current_area > best_area:
                    best_area = current_area
                    best_placements = list(current_placements)
        
        temp *= cooling_rate
        
        # Periodically reset temperature if stuck
        if iteration % 500 == 0:
            temp = max(temp, 0.1)

    return {'placements': best_placements}