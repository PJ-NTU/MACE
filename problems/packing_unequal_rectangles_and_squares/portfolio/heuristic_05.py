# MACE evolved heuristic 05/10 for problem: packing_unequal_rectangles_and_squares
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust packing heuristic combining adaptive multi-start construction
    with intensive iterative refinement.
    
    The strategy:
    1. Establish a strong baseline using area-decreasing greedy fill.
    2. Use a time-budgeted loop to explore the permutation space (prioritizing 
       different subsets of items) and apply local 'swap/refine' improvements.
    3. Periodically apply 'try_place_largest_unplaced' to fill gaps left by 
       the greedy construction.
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # 1. Baseline
    best_placements = tools['bottom_left_fill_decreasing']()
    best_placements = tools['try_place_largest_unplaced'](best_placements)
    best_score = len(best_placements)
    
    # 2. Iterative Improvement Loop
    # We maintain a time-budgeted search. 
    # We mix random permutations with 'swap_items' refinement.
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.95:
        iteration += 1
        
        # Strategy A: Multi-start with random permutations to discover new packing layouts
        # We bias the shuffle slightly towards area-descending to keep packing density high
        indices = list(range(n))
        if iteration % 2 == 0:
            # Perturbed area-descending order
            indices = sorted(range(n), key=lambda i: instance['items'][i][0] * instance['items'][i][1], reverse=True)
            for _ in range(max(1, n // 10)):
                i, j = random.sample(range(n), 2)
                indices[i], indices[j] = indices[j], indices[i]
        else:
            random.shuffle(indices)
            
        candidate = tools['bottom_left_pack'](indices)
        candidate = tools['try_place_largest_unplaced'](candidate)
        
        # Strategy B: Refinement on the current best
        if iteration % 5 == 0:
            # Apply swap_items to attempt to improve the best found so far
            refinement_time = (time_limit_s - (time.time() - start_time)) * 0.5
            if refinement_time > 0.05:
                candidate = tools['apply_swap_items'](best_placements, time_limit_s=refinement_time)
                candidate = tools['try_place_largest_unplaced'](candidate)

        # Update best
        current_score = len(candidate)
        if current_score > best_score:
            best_score = current_score
            best_placements = candidate
            
        # Early exit if we have packed everything
        if best_score == n:
            break
            
    return tools['placements_to_solution'](best_placements)