# MACE evolved heuristic 02/10 for problem: packing_unequal_rectangles_and_squares_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized packing heuristic using a combination of greedy construction
    and an Iterative Local Search (ILS) with adaptive perturbation.
    """
    start_time = time.time()
    n = instance['n']
    rotation_allowed = instance['rotation']
    
    # 1. Initialization: Greedy by area (descending)
    # This provides a strong baseline.
    current_placements = tools['greedy_by_area_first'](time_limit_s=time_limit_s * 0.2, grid=20)
    best_placements = list(current_placements)
    best_area = tools['total_area'](best_placements)
    
    # Pre-sort indices by area for better search efficiency
    items_with_area = sorted([(i, tools['item_area'](i)) for i in range(n)], key=lambda x: x[1], reverse=True)
    sorted_indices = [x[0] for x in items_with_area]
    
    rng = random.Random(42)
    
    # 2. ILS Loop
    # We maintain a 'current' solution and perform perturbations to escape local optima.
    while time.time() - start_time < time_limit_s * 0.95:
        # Clone current state
        work_placements = list(best_placements)
        
        # Perturbation: Remove a random subset of items (destructive)
        # This allows the greedy logic to re-pack effectively.
        num_remove = rng.randint(1, max(1, n // 4))
        for _ in range(num_remove):
            idx = rng.choice(sorted_indices)
            work_placements[idx] = (-1.0, -1.0, 0.0)
            
        # Re-fill: Greedy fill for missing items
        # Try to pack unpacked items in decreasing order of area
        for idx in sorted_indices:
            if work_placements[idx][0] == -1:
                # Try a few random orientations/positions
                possible_thetas = [0, 90] if rotation_allowed else [0]
                for _ in range(15):
                    theta = rng.choice(possible_thetas)
                    pos = tools['random_feasible_position'](idx, theta=theta, placements=work_placements, max_tries=30, rng=rng)
                    if pos:
                        x, y = pos
                        new_state = tools['try_place'](work_placements, idx, x, y, theta)
                        if new_state:
                            work_placements = new_state
                            break
                            
        # Evaluate
        current_area = tools['total_area'](work_placements)
        if current_area > best_area:
            # Final validation check
            is_valid, _ = tools['is_feasible']({'placements': work_placements})
            if is_valid:
                best_placements = work_placements
                best_area = current_area
        
        # Adaptive cooling: Optional, but here we keep full search intensity
        # because the search space is highly constrained.
        
    return {'placements': best_placements}