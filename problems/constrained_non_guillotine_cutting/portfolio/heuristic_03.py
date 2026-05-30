# MACE evolved heuristic 03/10 for problem: constrained_non_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A GRASP-inspired constructive heuristic for the constrained non-guillotine 
    cutting problem. Improved by focusing the randomized selection on 
    high-value pieces and explicitly ensuring min-demand satisfaction 
    during constructive phases.
    """
    start_time = time.time()
    
    pieces = instance['pieces']
    n_types = len(pieces)
    
    # Pre-calculate densities
    densities = []
    for i, p in enumerate(pieces):
        area = p['length'] * p['width']
        densities.append((i + 1, p['value'] / area if area > 0 else 0))
    
    best_placements = []
    best_score = -1.0
    
    # Main loop: Iterative refinement with time budget
    while time.time() - start_time < time_limit_s * 0.85:
        # Weighted Mutation: Use a stronger bias towards high-density pieces 
        # for initial placement, but force mandatory pieces first to ensure feasibility.
        
        # 1. Mandatory pieces first
        mandatory_order = [i+1 for i in range(n_types) if pieces[i]['min'] > 0]
        # 2. Remaining pieces sorted by density with stronger jitter
        remaining = [d for d in densities if pieces[d[0]-1]['min'] == 0]
        shuffled_remaining = sorted(
            remaining, 
            key=lambda x: x[1] * random.expovariate(0.5), 
            reverse=True
        )
        order = mandatory_order + [x[0] for x in shuffled_remaining]
        
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
            # Apply local improvement
            improved = tools['apply_swap_pieces'](current_placements, time_limit_s=0.1)
            final_placements = improved if improved else current_placements
            
            score = sum(pieces[p[0]-1]['value'] for p in final_placements)
            
            if score > best_score:
                best_score = score
                best_placements = final_placements
        
        if time.time() - start_time > time_limit_s * 0.95:
            break

    # Fallback to pure greedy if no valid solution found
    if not best_placements:
        greedy = tools['greedy_max_value_density'](allow_rotation=True)
        return {'placements': greedy}

    return {'placements': best_placements}