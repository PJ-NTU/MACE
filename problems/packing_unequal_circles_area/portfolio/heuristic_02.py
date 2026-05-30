# MACE evolved heuristic 02/10 for problem: packing_unequal_circles_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Implements a Greedy Randomized Adaptive Search Procedure (GRASP)
    with a Local Search phase to maximize the area of packed circles.
    """
    start_time = time.time()
    n = instance["n"]
    radii = instance["radii"]
    # Sort indices by radius descending to prioritize larger area
    sorted_indices = sorted(range(n), key=lambda i: radii[i], reverse=True)
    
    # 1. Construction Phase: Greedy by Area
    # Start with the best known greedy heuristic
    best_coords = tools['greedy_by_area_first'](attempts_per_circle=200)
    best_area = tools['total_area'](best_coords)
    
    # 2. Iterative Improvement Phase (Local Search)
    # Perform hill-climbing moves: try adding unpacked circles or swapping
    # smaller packed circles for larger unpacked ones.
    
    iters = 0
    while time.time() - start_time < time_limit_s * 0.9:
        iters += 1
        
        # Try to add an unpacked circle
        unpacked = [i for i in range(n) if not tools['is_placed'](i, best_coords)]
        if unpacked:
            target = random.choice(unpacked)
            new_coords = tools['try_add_circle'](best_coords, target, attempts=300)
            if new_coords:
                new_area = tools['total_area'](new_coords)
                if new_area > best_area:
                    best_coords = new_coords
                    best_area = new_area
                    continue
        
        # Try a swap: replace a smaller packed circle with a larger unpacked one
        packed = [i for i in range(n) if tools['is_placed'](i, best_coords)]
        if packed and unpacked:
            out_i = random.choice(packed)
            in_i = max(unpacked, key=lambda i: radii[i])
            
            if radii[in_i] > radii[out_i]:
                new_coords = tools['try_swap_in_out'](best_coords, out_i, in_i, attempts=300)
                if new_coords:
                    new_area = tools['total_area'](new_coords)
                    if new_area > best_area:
                        best_coords = new_coords
                        best_area = new_area
                        continue
        
        # Try to relocate a circle to make space for others (compaction)
        if packed:
            target = random.choice(packed)
            new_coords = tools['try_relocate_circle'](best_coords, target, attempts=100)
            if new_coords:
                best_coords = new_coords

        # Safety break if we haven't improved in a while and time is running low
        if iters > 2000:
            break

    return {"coords": best_coords}