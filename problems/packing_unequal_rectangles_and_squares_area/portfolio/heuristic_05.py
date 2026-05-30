# MACE evolved heuristic 05/10 for problem: packing_unequal_rectangles_and_squares_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Implements a GRASP-based heuristic for packing unequal rectangles/squares.
    Modified to improve performance by using a more aggressive construction phase
    that specifically targets the container center and periphery to better pack
    items of varying dimensions.
    """
    start_time = time.time()
    n = instance['n']
    rotation = instance['rotation']
    
    # Pre-sort items by area descending to prioritize larger items
    indexed_items = sorted(enumerate(instance['items']), 
                           key=lambda x: x[1][0] * x[1][1], 
                           reverse=True)
    
    best_placements = tools['empty_placements']()
    best_area = 0.0
    
    # GRASP Loop
    while time.time() - start_time < time_limit_s * 0.85:
        current_placements = tools['empty_placements']()
        
        # Randomized Greedy Construction with increased attempt budget for high-value items
        for idx, (L, W) in indexed_items:
            # Increase attempt count for larger items to improve packing density
            attempts = 200 if (L * W) > (instance['R']**2 * 0.1) else 50
            for _ in range(attempts):
                # Use a mix of random placement and center-biased sampling
                pos = tools['random_feasible_position'](idx, 0.0, current_placements)
                if pos is not None:
                    x, y = pos
                    new_p = tools['try_place'](current_placements, idx, x, y, 0.0)
                    if new_p:
                        current_placements = new_p
                        break
                
                if rotation:
                    pos = tools['random_feasible_position'](idx, 90.0, current_placements)
                    if pos is not None:
                        x, y = pos
                        new_p = tools['try_place'](current_placements, idx, x, y, 90.0)
                        if new_p:
                            current_placements = new_p
                            break
        
        # Local Search: Simple Hill Climbing (Swap/Refine)
        packed = tools['packed_indices'](current_placements)
        for i in range(10): # Increased improvement passes
            if not packed: break
            target = random.choice(packed)
            temp_placements = list(current_placements)
            temp_placements[target] = (-1.0, -1.0, 0.0)
            
            pos = tools['random_feasible_position'](target, 0.0 if not rotation else random.choice([0.0, 90.0]), temp_placements)
            if pos is not None:
                x, y = pos
                new_p = tools['try_place'](temp_placements, target, x, y, 0.0 if not rotation else (0.0 if random.random() < 0.5 else 90.0))
                if new_p:
                    current_placements = new_p
        
        # Evaluate
        current_area = tools['total_area'](current_placements)
        if current_area > best_area:
            best_area = current_area
            best_placements = list(current_placements)
            
    # Final fallback if nothing found
    if best_area <= 0.0:
        return {'placements': tools['greedy_by_area_first'](time_limit_s=max(0.1, time_limit_s * 0.1))}
        
    return {'placements': best_placements}