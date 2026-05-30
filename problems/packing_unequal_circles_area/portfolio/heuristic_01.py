# MACE evolved heuristic 01/10 for problem: packing_unequal_circles_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A GRASP-based metaheuristic for packing unequal circles.
    1. Construction: Greedy packing by area descending with randomized placement.
    2. Improvement: Iteratively try to swap packed circles with larger unpacked circles
       using local search (try_swap_in_out) and compaction (try_relocate_circle).
    """
    start_time = time.time()
    n = tools['num_circles']()
    radii = instance['radii']
    
    # Pre-sort indices by radius descending to prioritize larger circles
    sorted_indices = sorted(range(n), key=lambda i: radii[i], reverse=True)
    
    # 1. Construction: Greedy build
    best_coords = tools['greedy_by_area_first'](attempts_per_circle=200)
    best_score = tools['total_area'](best_coords)
    
    # 2. Local Search Loop
    # Keep track of time and improve the solution
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.9:
        iteration += 1
        
        # Identify current state
        placed = [i for i in range(n) if tools['is_placed'](i, best_coords)]
        unpacked = [i for i in range(n) if not tools['is_placed'](i, best_coords)]
        
        # Strategy A: Try to swap a small packed circle with a larger unpacked one
        if unpacked and placed:
            # Pick a random unpacked (prioritize larger ones) and a random packed (prioritize smaller ones)
            in_i = random.choice(unpacked)
            out_i = random.choice(placed)
            
            new_coords = tools['try_swap_in_out'](best_coords, out_i, in_i, attempts=100)
            if new_coords:
                new_score = tools['total_area'](new_coords)
                if new_score > best_score:
                    best_coords, best_score = new_coords, new_score
                    continue

        # Strategy B: Try to relocate a circle to open up space
        if placed:
            target = random.choice(placed)
            new_coords = tools['try_relocate_circle'](best_coords, target, attempts=100)
            if new_coords:
                # Relocation doesn't change area, but might allow future additions
                best_coords = new_coords
        
        # Strategy C: Try to add an unpacked circle
        if unpacked:
            target = random.choice(unpacked)
            new_coords = tools['try_add_circle'](best_coords, target, attempts=100)
            if new_coords:
                new_score = tools['total_area'](new_coords)
                if new_score > best_score:
                    best_coords, best_score = new_coords, new_score
        
        # Adaptive break: if we've done many iterations without improvement, 
        # restart construction to explore different parts of the search space
        if iteration % 50 == 0:
            current_coords = tools['greedy_pack_in_order'](
                sorted_indices, attempts_per_circle=100, rng_seed=random.randint(0, 10000)
            )
            current_score = tools['total_area'](current_coords)
            if current_score > best_score:
                best_coords, best_score = current_coords, current_score

    return {"coords": best_coords}