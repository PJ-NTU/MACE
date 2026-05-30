# MACE evolved heuristic 10/10 for problem: packing_unequal_rectangles_and_squares_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust packing heuristic using a combination of deterministic greedy
    construction and Adaptive Large Neighborhood Search (ALNS).
    """
    start_time = time.time()
    n = instance['n']
    rotation_allowed = instance['rotation']
    
    # Sort items by area descending
    items_with_area = sorted([(i, tools['item_area'](i)) for i in range(n)], 
                             key=lambda x: x[1], reverse=True)
    sorted_indices = [x[0] for x in items_with_area]
    
    rng = random.Random(42)
    
    # 1. Initial Construction
    current_placements = tools['greedy_by_area_first'](time_limit_s=time_limit_s * 0.15, grid=25)
    best_placements = list(current_placements)
    best_area = tools['total_area'](best_placements)
    
    # 2. Main Optimization Loop
    # We use a destructive-constructive approach (Ruin and Recreate)
    while time.time() - start_time < time_limit_s * 0.92:
        work_placements = list(best_placements)
        
        # Ruin: Remove a random number of items, biased towards removing smaller items
        # to allow space for larger ones, or removing a random chunk.
        num_remove = rng.randint(1, max(1, n // 3))
        # Choose strategy: 50% random, 50% target small items
        if rng.random() < 0.5:
            indices_to_remove = rng.sample(range(n), num_remove)
        else:
            indices_to_remove = sorted_indices[-num_remove:]
            
        for idx in indices_to_remove:
            work_placements[idx] = (-1.0, -1.0, 0.0)
            
        # Recreate: Greedy filling of empty slots
        # Strategy: iterate through potential items and try to place them
        # using a combination of random sampling and specific packing logic
        for idx in sorted_indices:
            if work_placements[idx][0] == -1:
                # Try high-probability orientations
                thetas = [0, 90] if rotation_allowed else [0]
                rng.shuffle(thetas)
                
                placed = False
                for theta in thetas:
                    # Use random_feasible_position for fast finding
                    pos = tools['random_feasible_position'](idx, theta=theta, 
                                                            placements=work_placements, 
                                                            max_tries=40, rng=rng)
                    if pos:
                        x, y = pos
                        new_state = tools['try_place'](work_placements, idx, x, y, theta)
                        if new_state:
                            work_placements = new_state
                            placed = True
                            break
                
        # Acceptance
        current_area = tools['total_area'](work_placements)
        if current_area > best_area:
            is_valid, _ = tools['is_feasible']({'placements': work_placements})
            if is_valid:
                best_placements = work_placements
                best_area = current_area
        
        # Periodic restart if stuck
        if rng.random() < 0.05:
            # Reset to a fresh greedy start to keep diversity
            new_start = tools['greedy_by_area_first'](time_limit_s=0.1, grid=10)
            if tools['total_area'](new_start) > best_area:
                best_placements = new_start
                best_area = tools['total_area'](best_placements)

    return {'placements': best_placements}