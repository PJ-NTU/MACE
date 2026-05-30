# MACE evolved heuristic 02/10 for problem: packing_unequal_rectangles_and_squares
import time
import math
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust heuristic for packing rectangles into a circular container.
    Combines bottom-left-fill with iterative local search and randomized
    restarts to maximize the number of packed items within the time limit.
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # 1. Start with a strong baseline: area-decreasing bottom-left fill
    best_placements = tools['bottom_left_fill_decreasing']()
    
    # Track the best number of items packed
    def count_packed(placements):
        return sum(1 for p in placements.values() if p[0] != -1)
    
    best_count = count_packed(best_placements)
    
    # 2. Local Search Improvement: 'remove-1, add-many'
    # We refine the initial greedy solution using the provided tool
    improved_placements = tools['apply_swap_items'](best_placements, time_limit_s=time_limit_s * 0.4)
    if count_packed(improved_placements) > best_count:
        best_placements = improved_placements
        best_count = count_packed(best_placements)
        
    # 3. Randomized Iterative Improvement
    # If time remains, perform randomized restarts with perturbed item sequences
    item_indices = list(range(n))
    while time.time() - start_time < time_limit_s * 0.9:
        # Generate a random permutation to explore different packing sequences
        random.shuffle(item_indices)
        current_placements = tools['bottom_left_pack'](item_indices)
        
        # Try to grow this specific layout
        current_placements = tools['try_place_largest_unplaced'](current_placements)
        
        # Apply local improvement to this new state
        current_placements = tools['apply_swap_items'](current_placements, time_limit_s=0.2)
        
        current_count = count_packed(current_placements)
        
        if current_count > best_count:
            best_placements = current_placements
            best_count = current_count
            
        # Break if we've packed everything
        if best_count == n:
            break
            
    # Final step: ensure we return the required dictionary format
    return tools['placements_to_solution'](best_placements)