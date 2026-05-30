# MACE evolved heuristic 06/10 for problem: packing_unequal_circles_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A GRASP-based metaheuristic for packing unequal circles.
    Modified: Strategy A now specifically focuses on swapping the smallest 
    packed circle with the largest unpacked circle to aggressively increase area.
    """
    start_time = time.time()
    n = tools['num_circles']()
    radii = instance['radii']
    
    # Pre-sort indices by radius descending
    sorted_indices = sorted(range(n), key=lambda i: radii[i], reverse=True)
    
    # 1. Construction: Greedy build
    best_coords = tools['greedy_by_area_first'](attempts_per_circle=200)
    best_score = tools['total_area'](best_coords)
    
    # 2. Local Search Loop
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.9:
        iteration += 1
        
        placed = [i for i in range(n) if tools['is_placed'](i, best_coords)]
        unpacked = [i for i in range(n) if not tools['is_placed'](i, best_coords)]
        
        # Strategy A: Targeted swap (Smallest Packed vs Largest Unpacked)
        if unpacked and placed:
            # Sort placed by radius ASC, unpacked by radius DESC
            placed_sorted = sorted(placed, key=lambda i: radii[i])
            unpacked_sorted = sorted(unpacked, key=lambda i: radii[i], reverse=True)
            
            in_i = unpacked_sorted[0]
            out_i = placed_sorted[0]
            
            if radii[in_i] > radii[out_i]:
                new_coords = tools['try_swap_in_out'](best_coords, out_i, in_i, attempts=150)
                if new_coords:
                    new_score = tools['total_area'](new_coords)
                    if new_score > best_score:
                        best_coords, best_score = new_coords, new_score
                        continue

        # Strategy B: Try to relocate a circle
        if placed:
            target = random.choice(placed)
            new_coords = tools['try_relocate_circle'](best_coords, target, attempts=100)
            if new_coords:
                best_coords = new_coords
        
        # Strategy C: Try to add an unpacked circle
        if unpacked:
            target = unpacked[0] # Try largest unpacked first
            new_coords = tools['try_add_circle'](best_coords, target, attempts=100)
            if new_coords:
                new_score = tools['total_area'](new_coords)
                if new_score > best_score:
                    best_coords, best_score = new_coords, new_score
        
        # Adaptive restart
        if iteration % 50 == 0:
            current_coords = tools['greedy_pack_in_order'](
                sorted_indices, attempts_per_circle=100, rng_seed=random.randint(0, 10000)
            )
            current_score = tools['total_area'](current_coords)
            if current_score > best_score:
                best_coords, best_score = current_coords, current_score

    return {"coords": best_coords}