# MACE evolved heuristic 09/10 for problem: packing_unequal_rectangles_and_squares_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized packing heuristic using a greedy-first construction combined
    with a focused Hill Climbing local search that prioritizes replacing
    smaller items with larger ones to maximize total area.
    """
    start_time = time.time()
    n = instance['n']
    rotation_allowed = instance['rotation']
    
    # 1. Initialization: Greedy by area (descending)
    # This provides a strong, high-density baseline.
    current_placements = tools['greedy_by_area_first'](time_limit_s=time_limit_s * 0.25, grid=20)
    
    best_placements = list(current_placements)
    best_area = tools['total_area'](best_placements)
    
    # Pre-sort indices by area descending to prioritize packing valuable items.
    items_by_area = sorted(range(n), key=lambda i: tools['item_area'](i), reverse=True)
    
    # 2. Local Search: Hill Climbing with "Swap-and-Squeeze"
    # Instead of random perturbation, we perform targeted neighborhood moves.
    # We attempt to free up space (by removing smaller items) to fit larger ones.
    
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit_s * 0.90:
        # Try to improve the current solution
        work_placements = list(best_placements)
        
        # Select a random subset of packed items to remove (destructive move)
        packed = tools['packed_indices'](work_placements)
        if not packed:
            # If empty, try to fill greedily
            for idx in items_by_area:
                pos = tools['random_feasible_position'](idx, theta=0, placements=work_placements, max_tries=20, rng=rng)
                if pos:
                    work_placements = tools['try_place'](work_placements, idx, pos[0], pos[1], 0) or work_placements
            best_placements = work_placements
            best_area = tools['total_area'](best_placements)
            continue

        # Remove 1-2 items to create a "hole"
        num_to_remove = rng.randint(1, min(len(packed), 3))
        for _ in range(num_to_remove):
            idx = rng.choice(packed)
            work_placements[idx] = (-1.0, -1.0, 0.0)
            
        # Attempt to fill the hole with larger items not currently packed
        # This is a greedy hill-climbing step
        for idx in items_by_area:
            if work_placements[idx][0] == -1:
                possible_thetas = [0, 90] if rotation_allowed else [0]
                # Try placing in the vacated space or elsewhere
                for _ in range(25):
                    theta = rng.choice(possible_thetas)
                    pos = tools['random_feasible_position'](idx, theta=theta, placements=work_placements, max_tries=40, rng=rng)
                    if pos:
                        new_state = tools['try_place'](work_placements, idx, pos[0], pos[1], theta)
                        if new_state:
                            work_placements = new_state
                            break
        
        # Evaluate improvement
        current_area = tools['total_area'](work_placements)
        if current_area > best_area:
            # Verify feasibility before accepting
            is_valid, _ = tools['is_feasible']({'placements': work_placements})
            if is_valid:
                best_placements = work_placements
                best_area = current_area
                
    return {'placements': best_placements}