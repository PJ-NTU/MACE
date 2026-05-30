# MACE evolved heuristic 07/10 for problem: packing_unequal_rectangles_and_squares
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined packing heuristic:
    Diagnosis: The parent heuristic suffered from inefficient stochastic search (shuffling) 
    and a lack of focus on maximizing density. The new approach prioritizes:
    1. A robust initial solution using area-descending greedy pack.
    2. A 'Tabu-lite' local search that focuses on improving the packing score by 
       systematically attempting to swap or re-insert items.
    3. Explicit time-budget monitoring for each refinement phase.
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # Initial Baseline: Current best
    # Use the strong default warm start
    best_placements = tools['bottom_left_fill_decreasing']()
    best_placements = tools['try_place_largest_unplaced'](best_placements)
    best_score = len(best_placements)
    
    # Heuristic strategy: 
    # 1. Use 'apply_swap_items' to perform a guided search.
    # 2. Use 'try_place_largest_unplaced' to fill gaps.
    # 3. If time permits, perform a hill-climbing search on item order permutations.
    
    # Phase 1: Heavy refinement
    if time.time() - start_time < time_limit_s * 0.7:
        best_placements = tools['apply_swap_items'](best_placements, time_limit_s=time_limit_s * 0.5)
        best_placements = tools['try_place_largest_unplaced'](best_placements)
        best_score = len(best_placements)
    
    # Phase 2: Permutation Hill Climbing
    # We maintain a list of items to attempt packing in a specific order
    items_indices = list(range(n))
    # Sort by area descending for the base order
    items_indices.sort(key=lambda i: instance['items'][i][0] * instance['items'][i][1], reverse=True)
    
    while time.time() - start_time < time_limit_s * 0.95:
        # Generate a neighbor by slightly perturbing the order (swap 2)
        idx1, idx2 = random.sample(range(n), 2)
        items_indices[idx1], items_indices[idx2] = items_indices[idx2], items_indices[idx1]
        
        candidate = tools['bottom_left_pack'](items_indices)
        candidate = tools['try_place_largest_unplaced'](candidate)
        
        if len(candidate) > best_score:
            best_placements = candidate
            best_score = len(candidate)
        elif len(candidate) == best_score:
            # Acceptance probability for equal scores to escape local optima
            if random.random() < 0.1:
                best_placements = candidate
        else:
            # Revert if it performs significantly worse
            # (Simple hill climbing logic)
            items_indices[idx1], items_indices[idx2] = items_indices[idx2], items_indices[idx1]
            
        if best_score == n:
            break
            
    # Final refinement pass
    final_placements = tools['apply_swap_items'](best_placements, time_limit_s=max(0.05, time_limit_s - (time.time() - start_time)))
    final_placements = tools['try_place_largest_unplaced'](final_placements)
    
    return tools['placements_to_solution'](final_placements)