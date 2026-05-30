# MACE evolved heuristic 02/10 for problem: constrained_non_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A GRASP-inspired constructive heuristic for the constrained non-guillotine 
    cutting problem. It uses value-density greedy packing combined with 
    randomized selection to explore the solution space.
    """
    start_time = time.time()
    
    # Extract problem data
    pieces = instance['pieces']
    n_types = len(pieces)
    
    # Pre-calculate densities for greedy ordering
    # Density = value / (length * width)
    densities = []
    for i, p in enumerate(pieces):
        area = p['length'] * p['width']
        densities.append((i + 1, p['value'] / area if area > 0 else 0))
    
    best_placements = []
    best_score = -1.0
    
    # Main loop: Iterative refinement with time budget
    while time.time() - start_time < time_limit_s * 0.85:
        # Sort by density, but add a small amount of jitter (GRASP)
        # to explore different packing sequences.
        shuffled_densities = sorted(
            densities, 
            key=lambda x: x[1] * random.uniform(0.7, 1.3), 
            reverse=True
        )
        order = [x[0] for x in shuffled_densities]
        
        current_placements = []
        counts = [0] * n_types
        
        # Try to fill the board with the current randomized order
        for t in order:
            while counts[t-1] < pieces[t-1]['max']:
                placement = tools['try_place_at_corner'](current_placements, t, allow_rotation=True)
                if placement:
                    current_placements.append(placement)
                    counts[t-1] += 1
                else:
                    break
        
        # Check feasibility regarding MIN constraints
        is_valid = True
        for i in range(n_types):
            if counts[i] < pieces[i]['min']:
                is_valid = False
                break
        
        if is_valid:
            # Calculate score
            score = sum(pieces[p[0]-1]['value'] for p in current_placements)
            
            # Apply local improvement
            improved = tools['apply_swap_pieces'](current_placements, time_limit_s=0.1)
            if improved:
                current_placements = improved
                score = sum(pieces[p[0]-1]['value'] for p in current_placements)
            
            if score > best_score:
                best_score = score
                best_placements = current_placements
        
        # Safety break
        if time.time() - start_time > time_limit_s * 0.95:
            break

    # Fallback to pure greedy if no valid solution found
    if not best_placements:
        greedy = tools['greedy_max_value_density'](allow_rotation=True)
        return {'placements': greedy}

    return {'placements': best_placements}