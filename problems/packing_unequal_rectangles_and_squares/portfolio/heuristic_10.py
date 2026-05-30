# MACE evolved heuristic 10/10 for problem: packing_unequal_rectangles_and_squares
import time
import math
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved heuristic for packing rectangles into a circular container.
    The modification replaces the pure-random shuffle with a temperature-based
    simulated annealing approach to the item ordering sequence to better explore
    the combinatorial space.
    """
    start_time = time.time()
    
    # 1. Initialization
    current_placements = tools['bottom_left_fill_decreasing']()
    
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

    # 2. Local Search
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.5:
        current_placements = tools['apply_swap_items'](current_placements, time_limit_s=remaining_time * 0.8)

    # 3. Simulated Annealing for sequence ordering
    n = tools['n_items']()
    items_areas = []
    for i in range(n):
        l, w = tools['item_dims'](i)
        items_areas.append((l * w, i))
    
    base_order = [x[1] for x in sorted(items_areas, key=lambda x: x[0], reverse=True)]
    current_order = list(base_order)
    
    temp = 1.0
    while time.time() - start_time < time_limit_s * 0.95:
        # Mutate: swap two random indices with probability dependent on temperature
        candidate_order = list(current_order)
        i, j = random.sample(range(n), 2)
        candidate_order[i], candidate_order[j] = candidate_order[j], candidate_order[i]
        
        candidate_placements = tools['bottom_left_pack'](candidate_order)
        candidate_placements = polish(candidate_placements)
        
        score_diff = len(candidate_placements) - len(current_placements)
        
        # Metropolis acceptance criterion
        if score_diff > 0 or (temp > 0 and math.exp(score_diff / temp) > random.random()):
            current_order = candidate_order
            if len(candidate_placements) > len(current_placements):
                current_placements = candidate_placements
        
        # Cool down
        temp *= 0.99
            
    return tools['placements_to_solution'](current_placements)