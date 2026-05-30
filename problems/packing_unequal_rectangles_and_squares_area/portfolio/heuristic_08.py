# MACE evolved heuristic 08/10 for problem: packing_unequal_rectangles_and_squares_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A dispatcher-style heuristic that selects between an Iterative Local Search (ILS)
    and a Simulated Annealing (SA) meta-heuristic based on the instance's 
    'Packing Density'. 
    
    Hypothesis: 
    - High-density instances (small total area relative to container) are 
      constrained by geometry, requiring aggressive constructive gap-filling (ILS).
    - Low-density instances allow more freedom, where SA's stochastic exploration 
      discovers better non-greedy arrangements.
    """
    start_time = time.time()
    n = instance['n']
    R = instance['R']
    items = instance['items']
    rotation = instance['rotation']
    
    container_area = math.pi * (R ** 2)
    total_item_area = sum(l * w for l, w in items)
    
    # Heuristic: Density ratio. If items cover > 50% of area, it's 'tight'.
    if (total_item_area / container_area) > 0.5:
        return _solve_ils(instance, tools, time_limit_s)
    else:
        return _solve_sa(instance, tools, time_limit_s)

def _solve_ils(instance, tools, time_limit_s):
    start_time = time.time()
    n = instance['n']
    rotation = instance['rotation']
    
    best_placements = tools['greedy_by_area_first'](time_limit_s=time_limit_s * 0.25, grid=20)
    best_area = tools['total_area'](best_placements)
    
    # Sort indices by area to prioritize removing smaller items during perturbation
    items_with_area = sorted([(i, tools['item_area'](i)) for i in range(n)], key=lambda x: x[1], reverse=True)
    sorted_indices = [x[0] for x in items_with_area]
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit_s * 0.90:
        work_placements = list(best_placements)
        # Perturb: Remove random subset
        num_remove = rng.randint(1, max(1, n // 4))
        for _ in range(num_remove):
            idx = rng.choice(sorted_indices)
            work_placements[idx] = (-1.0, -1.0, 0.0)
            
        # Reconstruct greedy
        for idx in sorted_indices:
            if work_placements[idx][0] == -1:
                possible_thetas = [0, 90] if rotation else [0]
                # Small local search for placement
                for _ in range(5):
                    theta = rng.choice(possible_thetas)
                    pos = tools['random_feasible_position'](idx, theta=theta, placements=work_placements, max_tries=30, rng=rng)
                    if pos:
                        new_state = tools['try_place'](work_placements, idx, pos[0], pos[1], theta)
                        if new_state:
                            work_placements = new_state
                            break
                            
        current_area = tools['total_area'](work_placements)
        if current_area > best_area:
            is_valid, _ = tools['is_feasible']({'placements': work_placements})
            if is_valid:
                best_placements = work_placements
                best_area = current_area
    return {'placements': best_placements}

def _solve_sa(instance, tools, time_limit_s):
    start_time = time.time()
    n = instance['n']
    rotation = instance['rotation']
    best_placements = tools['greedy_by_area_first'](time_limit_s=time_limit_s * 0.25, grid=20)
    best_area = tools['total_area'](best_placements)
    current_placements = list(best_placements)
    current_area = best_area
    
    # SA params
    temp = 10.0
    cooling_rate = 0.9995
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit_s * 0.90:
        idx = rng.randint(0, n - 1)
        new_placements = list(current_placements)
        
        # Decide: Add, Remove, or Swap
        op = rng.random()
        if op < 0.4: # Remove
            new_placements[idx] = (-1.0, -1.0, 0.0)
        else: # Try place
            theta = rng.choice([0, 90]) if rotation else 0
            pos = tools['random_feasible_position'](idx, theta, current_placements, max_tries=50, rng=rng)
            if pos:
                new_placements[idx] = (pos[0], pos[1], theta)
        
        is_valid, _ = tools['is_feasible']({'placements': new_placements})
        if is_valid:
            new_area = tools['total_area'](new_placements)
            delta = new_area - current_area
            # Accept if better or random chance
            if delta > 0 or (temp > 0.01 and math.exp(delta / (temp + 1e-9)) > rng.random()):
                current_placements = new_placements
                current_area = new_area
                if current_area > best_area:
                    best_area = current_area
                    best_placements = list(current_placements)
        temp *= cooling_rate
    return {'placements': best_placements}