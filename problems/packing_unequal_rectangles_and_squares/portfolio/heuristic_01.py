# MACE evolved heuristic 01/10 for problem: packing_unequal_rectangles_and_squares
import time
import math
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust heuristic for packing rectangles into a circular container.
    Combines bottom-left-fill with randomized local search and improvement cycles.
    """
    start_time = time.time()
    
    # 1. Initialization: Get the strong default warm start
    # bottom_left_fill_decreasing is a powerful O(N^2) greedy baseline.
    current_placements = tools['bottom_left_fill_decreasing']()
    
    # 2. Refinement loop: Try to pack as many of the remaining items as possible.
    # We repeatedly try to insert the largest unplaced items.
    def polish(placements):
        improved = True
        while improved:
            if time.time() - start_time > time_limit_s * 0.4:
                break
            new_placements = tools['try_place_largest_unplaced'](placements)
            if len(new_placements) > len(placements):
                placements = new_placements
            else:
                improved = False
        return placements

    current_placements = polish(current_placements)

    # 3. Local Search: Iteratively remove small items to create space for larger ones
    # or to force the algorithm to re-evaluate the packing.
    # We use the provided apply_swap_items to perform remove-1, add-many improvements.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.5:
        current_placements = tools['apply_swap_items'](current_placements, time_limit_s=remaining_time * 0.8)

    # 4. Randomized Hill Climbing (if time permits)
    # Shuffle item order and try different greedy packing orders to find better configurations.
    n = tools['n_items']()
    while time.time() - start_time < time_limit_s * 0.95:
        # Create a random permutation of items to try a different greedy path
        indices = list(range(n))
        random.shuffle(indices)
        
        candidate_placements = tools['bottom_left_pack'](indices)
        candidate_placements = polish(candidate_placements)
        
        if len(candidate_placements) > len(current_placements):
            current_placements = candidate_placements
        else:
            # Maybe keep it if it's the same size but potentially better for future additions
            # but for this objective, size is all that matters.
            pass
            
    # 5. Format and return
    return tools['placements_to_solution'](current_placements)