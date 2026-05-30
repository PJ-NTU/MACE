# MACE evolved heuristic 03/10 for problem: packing_unequal_rectangles_and_squares_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized packing heuristic employing a multi-start GRASP-like approach.
    It combines deterministic greedy construction with randomized local search 
    perturbations, focusing on high-density packing and efficient time usage.
    """
    start_time = time.time()
    n = instance['n']
    rotation_allowed = instance['rotation']
    
    # Pre-calculate area for all items to guide the packing priority
    items_info = []
    for i in range(n):
        L, W = instance['items'][i]
        items_info.append((i, L * W))
    
    # Sort by area descending
    sorted_items = sorted(items_info, key=lambda x: x[1], reverse=True)
    sorted_indices = [x[0] for x in sorted_items]
    
    # 1. Baseline Construction
    best_placements = tools['greedy_by_area_first'](time_limit_s=max(0.1, time_limit_s * 0.15), grid=25)
    best_area = tools['total_area'](best_placements)
    
    rng = random.Random(42)
    
    # 2. Iterative Local Search with Adaptive Perturbation
    # Strategy: Start from the best found so far and perform "destructive" 
    # re-packing to explore the neighborhood of the solution.
    while time.time() - start_time < time_limit_s * 0.95:
        # Clone current best
        work_placements = list(best_placements)
        
        # Perturbation: Remove a random percentage of packed items
        packed = tools['packed_indices'](work_placements)
        if not packed:
            # Fallback if empty
            work_placements = tools['empty_placements']()
        else:
            # Remove 1 to 3 items to open space for better configurations
            num_to_remove = rng.randint(1, min(len(packed), 3))
            to_remove = rng.sample(packed, num_to_remove)
            for idx in to_remove:
                work_placements[idx] = (-1.0, -1.0, 0.0)
        
        # Re-fill: Greedy fill with randomized noise
        # We process items in a shuffled order that biases toward large items
        current_indices = list(sorted_indices)
        # Add slight stochasticity to the order
        if rng.random() < 0.3:
            rng.shuffle(current_indices)
            
        for idx in current_indices:
            if work_placements[idx][0] == -1:
                # Try placing in random valid positions
                possible_thetas = [0, 90] if rotation_allowed else [0]
                # High effort for small number of items
                for theta in possible_thetas:
                    pos = tools['random_feasible_position'](idx, theta=theta, placements=work_placements, max_tries=100, rng=rng)
                    if pos:
                        x, y = pos
                        new_state = tools['try_place'](work_placements, idx, x, y, theta)
                        if new_state:
                            work_placements = new_state
                            break
        
        # Acceptance: Evaluate and update
        current_area = tools['total_area'](work_placements)
        if current_area > best_area:
            # Double check feasibility
            is_valid, _ = tools['is_feasible']({'placements': work_placements})
            if is_valid:
                best_area = current_area
                best_placements = work_placements
        
        # Early exit if we reach a high-quality bound
        if best_area >= tools['container_area']() * 0.98:
            break
            
    return {'placements': best_placements}