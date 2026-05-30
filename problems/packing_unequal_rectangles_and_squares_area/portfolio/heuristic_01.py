# MACE evolved heuristic 01/10 for problem: packing_unequal_rectangles_and_squares_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Implements a GRASP-based heuristic (Greedy Randomized Adaptive Search Procedure)
    for the rectangle packing problem.
    """
    start_time = time.time()
    n = instance['n']
    items = instance['items']
    rotation_allowed = instance['rotation']
    
    # Sort items by area descending (Heuristic: pack larger items first)
    indexed_items = sorted(enumerate(items), key=lambda x: x[1][0] * x[1][1], reverse=True)
    
    best_placements = tools['empty_placements']()
    best_area = 0.0
    
    # Run iterations until time limit
    while time.time() - start_time < time_limit_s * 0.9:
        current_placements = tools['empty_placements']()
        
        # Randomized Greedy Construction
        for idx, (L, W) in indexed_items:
            # Try to find a valid position using randomized sampling
            # We use a mix of random spots and potential "tighter" spots
            possible_thetas = [0, 90] if rotation_allowed else [0]
            
            # Try a few attempts to place this specific item
            placed = False
            for _ in range(50):
                theta = random.choice(possible_thetas)
                pos = tools['random_feasible_position'](idx, theta=theta, placements=current_placements, max_tries=100)
                if pos:
                    new_placements = tools['try_place'](current_placements, idx, pos[0], pos[1], theta)
                    if new_placements:
                        current_placements = new_placements
                        placed = True
                        break
            
            if not placed:
                continue
        
        # Evaluate current construction
        current_area = tools['total_area'](current_placements)
        if current_area > best_area:
            best_area = current_area
            best_placements = list(current_placements)
            
        # Early exit if we reached the container capacity (theoretical)
        if best_area >= tools['container_area']() * 0.95:
            break
            
    return {'placements': best_placements}