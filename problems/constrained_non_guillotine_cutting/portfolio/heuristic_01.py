# MACE evolved heuristic 01/10 for problem: constrained_non_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Constrained Non-Guillotine Cutting Problem using a 
    Randomized Bottom-Left Greedy approach with iterated improvement.
    """
    start_time = time.time()
    
    # Extract problem data
    pieces = instance['pieces']
    n_types = len(pieces)
    
    # Prepare sorting criteria: value density (value / area)
    type_indices = list(range(1, n_types + 1))
    
    def get_piece_density(t):
        p = pieces[t-1]
        area = p['length'] * p['width']
        return p['value'] / area if area > 0 else 0

    best_placements = []
    best_score = -1.0
    
    # Main loop: Iterative randomized construction
    while (time.time() - start_time) < (time_limit_s * 0.8):
        # Randomized priority: sort by density with slight noise
        current_order = sorted(type_indices, key=lambda t: get_piece_density(t) * random.uniform(0.8, 1.2), reverse=True)
        
        placements = []
        counts = [0] * n_types
        
        # Constructive phase
        for t in current_order:
            while counts[t-1] < pieces[t-1]['max']:
                placement = tools['try_place_at_corner'](placements, t, allow_rotation=True)
                if placement:
                    placements.append(placement)
                    counts[t-1] += 1
                else:
                    break
        
        # Check minimum demand constraint
        feasible = True
        for i in range(n_types):
            if counts[i] < pieces[i]['min']:
                feasible = False
                break
        
        if feasible:
            # Improvement phase: try to swap pieces
            improved = tools['apply_swap_pieces'](placements, time_limit_s=0.1)
            
            # Re-verify feasibility and score
            if improved:
                current_placements = improved
            else:
                current_placements = placements
                
            score = 0
            for p in current_placements:
                score += pieces[p[0]-1]['value']
            
            if score > best_score:
                best_score = score
                best_placements = list(current_placements)
                
    # Fallback: If no feasible solution found, attempt a minimal greedy fill
    if not best_placements:
        placements = tools['greedy_max_value_density'](allow_rotation=True)
        # Verify min constraints for the fallback
        counts = tools['used_count_per_type'](placements)
        if all(counts[i] >= pieces[i]['min'] for i in range(n_types)):
            best_placements = placements
        else:
            # Return empty or partial if strict min constraints cannot be met
            return {'placements': []}

    return {'placements': best_placements}