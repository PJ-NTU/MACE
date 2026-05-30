# MACE evolved heuristic 06/10 for problem: packing_unequal_rectangles_and_squares
import time
import math
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A dispatcher-style heuristic that uses instance-aware feature selection:
    1. If the instance has a high count of small items relative to container size 
       (low density), it prioritizes Global-Orientation (B-style) to explore 
       the continuous space.
    2. If the instance is dense or has large, irregular items, it uses the 
       Priority-Permutation Simulated Annealing (A-style) which is more 
       effective at packing structured/constrained layouts.
    """
    start_time = time.time()
    n = tools['n_items']()
    cx, cy, R = tools['container_dims']()
    
    # Calculate packing density metric
    total_area = sum(item[0] * item[1] for item in instance['items'])
    container_area = math.pi * (R ** 2)
    density = total_area / container_area if container_area > 0 else 1.0
    
    # Dispatcher Logic:
    # Dense instances (density > 0.5) are better handled by ordering-based SA (A-style).
    # Sparse/Large-item instances are better handled by coordinate-based exploration (B-style).
    if density > 0.5:
        # Strategy A: SA over Permutations
        current_perm = list(range(n))
        random.shuffle(current_perm)
        
        best_placements = tools['bottom_left_pack'](current_perm)
        best_score = len(best_placements)
        
        temp = 1.0
        while time.time() - start_time < time_limit_s * 0.8:
            neighbor_perm = current_perm[:]
            idx1, idx2 = random.sample(range(n), 2)
            neighbor_perm[idx1], neighbor_perm[idx2] = neighbor_perm[idx2], neighbor_perm[idx1]
            
            placements = tools['bottom_left_pack'](neighbor_perm)
            score = len(placements)
            
            delta = score - best_score
            if delta >= 0 or (temp > 0 and random.random() < math.exp(delta / temp)):
                current_perm = neighbor_perm
                if score > best_score:
                    best_score = score
                    best_placements = placements
            
            temp *= 0.99
            if temp < 1e-3: temp = 0.5
            
        final_placements = tools['try_place_largest_unplaced'](best_placements)
    else:
        # Strategy B: Stochastic Coordinate Hill Climbing
        current_placements = tools['bottom_left_fill_decreasing']()
        
        while time.time() - start_time < time_limit_s * 0.8:
            idx = random.choice(range(n))
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(0, R * 0.8)
            new_x = cx + dist * math.cos(angle)
            new_y = cy + dist * math.sin(angle)
            new_theta = 90.0 if (instance.get('rotation') and random.random() > 0.5) else 0.0
            
            if tools['can_fit_at'](current_placements, idx, new_x, new_y, new_theta):
                current_placements[idx] = (new_x, new_y, new_theta)
            
            if random.random() < 0.05:
                current_placements = tools['try_place_largest_unplaced'](current_placements)
                
        final_placements = tools['try_place_largest_unplaced'](current_placements)

    # Convert to required format
    full_placements = {}
    for i in range(n):
        if i in final_placements:
            full_placements[i] = final_placements[i]
        else:
            full_placements[i] = (-1, -1, 0)
            
    return tools['placements_to_solution'](full_placements)