# MACE evolved heuristic 09/10 for problem: packing_unequal_rectangles_and_squares
import time
import math
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized hybrid heuristic that combines the strengths of greedy area-based
    prioritization, local search (remove-add), and stochastic perturbation to
    maximize the number of packed items in a circular container.
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # 1. Initialize with the strongest known deterministic greedy approach
    # (Area-decreasing is a very strong baseline for packing problems)
    best_placements = tools['bottom_left_fill_decreasing']()
    
    def count_packed(placements):
        return sum(1 for p in placements.values() if p[0] != -1)
    
    best_count = count_packed(best_placements)
    
    # 2. Refine the initial solution using the robust 'apply_swap_items' tool
    # This addresses gaps created by the initial greedy pass
    improved_placements = tools['apply_swap_items'](best_placements, time_limit_s=max(0.1, time_limit_s * 0.2))
    if count_packed(improved_placements) > best_count:
        best_placements = improved_placements
        best_count = count_packed(best_placements)
        
    # 3. Iterative Stochastic Search
    # Use a priority-based exploration strategy. Unlike h_a's pure SA, we combine
    # the structure of h_b's local search with randomized perturbations.
    item_indices = list(range(n))
    
    # Sort by area descending as the base for perturbation
    # (Pre-calculate areas once)
    areas = []
    for i in range(n):
        l, w = tools['item_dims'](i)
        areas.append((l * w, i))
    areas.sort(key=lambda x: x[0], reverse=True)
    sorted_indices = [x[1] for x in areas]
    
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.95:
        iteration += 1
        
        # Every 5 iterations, use a fully random sequence to escape local minima.
        # Otherwise, use a "shuffled" version of the sorted sequence (biasing towards large items)
        if iteration % 5 == 0:
            current_perm = list(range(n))
            random.shuffle(current_perm)
        else:
            # Swap a few items in the sorted list to explore the neighborhood
            current_perm = sorted_indices[:]
            for _ in range(max(1, n // 10)):
                i, j = random.sample(range(n), 2)
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                
        # Generate packing
        current_placements = tools['bottom_left_pack'](current_perm)
        
        # Polish: try to insert any remaining unplaced items
        current_placements = tools['try_place_largest_unplaced'](current_placements)
        
        # Apply local improvement to the new state
        # Use a smaller time budget for local search to allow more global restarts
        current_placements = tools['apply_swap_items'](current_placements, time_limit_s=0.1)
        
        current_count = count_packed(current_placements)
        
        if current_count > best_count:
            best_placements = current_placements
            best_count = current_count
            
        # If we have packed everything, we are done
        if best_count == n:
            break
            
    return tools['placements_to_solution'](best_placements)