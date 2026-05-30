# MACE evolved heuristic 03/10 for problem: packing_unequal_rectangles_and_squares
import time
import math
import random
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A heuristic solver using Simulated Annealing with a state representation 
    focused on packing priority rather than fixed greedy construction.
    
    Unlike the portfolio, which relies heavily on deterministic bottom-left 
    greedy fills, this heuristic uses a stochastic search over the priority 
    permutation space, accepting worse moves occasionally to escape local 
    optima (Simulated Annealing).
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # Current state: a permutation of item indices to feed the bottom-left packer
    current_perm = list(range(n))
    random.shuffle(current_perm)
    
    def get_score(perm):
        placements = tools['bottom_left_pack'](perm)
        return len(placements), placements

    current_score, current_placements = get_score(current_perm)
    best_placements = current_placements
    best_score = current_score
    
    # Simulated Annealing parameters
    temp = 1.0
    cooling_rate = 0.95
    
    while time.time() - start_time < time_limit_s * 0.9:
        # Generate neighbor by swapping two random items in the priority sequence
        neighbor_perm = current_perm[:]
        idx1, idx2 = random.sample(range(n), 2)
        neighbor_perm[idx1], neighbor_perm[idx2] = neighbor_perm[idx2], neighbor_perm[idx1]
        
        neighbor_score, neighbor_placements = get_score(neighbor_perm)
        
        # Acceptance criteria
        delta = neighbor_score - current_score
        if delta > 0 or (temp > 0 and random.random() < math.exp(delta / temp)):
            current_perm = neighbor_perm
            current_score = neighbor_score
            current_placements = neighbor_placements
            
            if current_score > best_score:
                best_score = current_score
                best_placements = current_placements
        
        # Cool down
        temp *= cooling_rate
        
        # Occasional "kick" if stuck at low temp
        if temp < 1e-4:
            temp = 0.5
            random.shuffle(current_perm)
            
    # Final refinement: attempt to force-fit remaining items into gaps
    final_placements = tools['try_place_largest_unplaced'](best_placements)
    
    return tools['placements_to_solution'](final_placements)