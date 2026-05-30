# MACE evolved heuristic 08/10 for problem: packing_unequal_rectangles_and_squares
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized packing heuristic using a greedy-randomized multi-start approach
    with aggressive local refinement.
    
    The strategy:
    1. Baseline: Uses area-decreasing greedy fill.
    2. Adaptive Search: Alternates between:
       a) Perturbed greedy construction (biased toward area-descending order).
       b) Intensive local search using `apply_swap_items` and `try_place_largest_unplaced`.
    3. Efficiency: Avoids complex state management (like SA temperatures) in favor of 
       highly effective primitive-based local search iterations.
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # Calculate areas to facilitate smart sorting
    items = instance['items']
    areas = [l * w for l, w in items]
    area_desc_indices = sorted(range(n), key=lambda i: areas[i], reverse=True)
    
    # Initial baseline
    best_placements = tools['bottom_left_fill_decreasing']()
    best_placements = tools['try_place_largest_unplaced'](best_placements)
    best_score = len(best_placements)
    
    # Track time and iterations
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.95:
        iteration += 1
        
        # 1. Perturbed Construction
        # Use a "shaken" area-descending order to explore different packing configurations
        if iteration % 3 != 0:
            indices = list(area_desc_indices)
            # Apply a small number of swaps to explore near-optimal permutations
            num_swaps = random.randint(1, max(1, n // 5))
            for _ in range(num_swaps):
                i, j = random.sample(range(n), 2)
                indices[i], indices[j] = indices[j], indices[i]
            
            candidate = tools['bottom_left_pack'](indices)
        else:
            # Random permutation for broader exploration
            indices = list(range(n))
            random.shuffle(indices)
            candidate = tools['bottom_left_pack'](indices)
            
        candidate = tools['try_place_largest_unplaced'](candidate)
        
        # 2. Local Improvement
        # Periodically apply the powerful swap_items heuristic to the best found
        if iteration % 4 == 0:
            remaining = time_limit_s - (time.time() - start_time)
            if remaining > 0.1:
                candidate = tools['apply_swap_items'](candidate, time_limit_s=remaining * 0.3)
                candidate = tools['try_place_largest_unplaced'](candidate)
        
        # Update best
        current_score = len(candidate)
        if current_score > best_score:
            best_score = current_score
            best_placements = candidate
            
        # Early termination
        if best_score == n:
            break
            
    return tools['placements_to_solution'](best_placements)