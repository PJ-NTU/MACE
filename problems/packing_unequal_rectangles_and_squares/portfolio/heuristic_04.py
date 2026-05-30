# MACE evolved heuristic 04/10 for problem: packing_unequal_rectangles_and_squares
import time
import math
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Modified heuristic: Uses a weighted-randomized greedy construction strategy
    to prioritize packing smaller items once larger items have been placed, 
    improving density in the leftover gaps of the circular container.
    """
    start_time = time.time()
    n = tools['n_items']()
    items = instance['items']
    # Pre-calculate areas for weighted selection
    areas = [l * w for l, w in items]
    
    # 1. Start with a strong baseline
    best_placements = tools['bottom_left_fill_decreasing']()
    
    def count_packed(placements):
        return sum(1 for p in placements.values() if p[0] != -1)
    
    best_count = count_packed(best_placements)
    
    # 2. Local Search Improvement
    improved_placements = tools['apply_swap_items'](best_placements, time_limit_s=time_limit_s * 0.3)
    if count_packed(improved_placements) > best_count:
        best_placements = improved_placements
        best_count = count_packed(best_placements)
        
    # 3. Randomized Iterative Improvement with Weighted Mutation
    # Instead of purely random, we bias selection towards smaller items 
    # to fill the gaps that 'decreasing' area strategies often leave.
    item_indices = list(range(n))
    
    while time.time() - start_time < time_limit_s * 0.9:
        # Weighted shuffle: prioritize small items (1/area) to fill gaps
        # This acts as a mutation that complements the 'decreasing' baseline
        weights = [1.0 / (a + 1e-6) for a in areas]
        shuffled = sorted(item_indices, key=lambda i: random.random() * weights[i], reverse=True)
        
        current_placements = tools['bottom_left_pack'](shuffled)
        
        # Try to grow this specific layout
        current_placements = tools['try_place_largest_unplaced'](current_placements)
        
        # Apply local improvement
        current_placements = tools['apply_swap_items'](current_placements, time_limit_s=0.1)
        
        current_count = count_packed(current_placements)
        
        if current_count > best_count:
            best_placements = current_placements
            best_count = current_count
            
        if best_count == n:
            break
            
    return tools['placements_to_solution'](best_placements)