# MACE evolved heuristic 08/10 for problem: constrained_non_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved heuristic for 2D non-guillotine cutting.
    
    Diagnosis of parent:
    - The dual-strategy approach was unnecessarily complex and non-adaptive.
    - The 'apply_swap_pieces' utility is powerful but was under-utilized.
    - The constructive phase relied too heavily on fixed greedy orders.
    
    Redesign:
    - Use a Randomized Variable Neighborhood Search (RVNS) skeleton.
    - Constructive phase: Randomized greedy packing based on density * jitter.
    - Improvement phase: Iterative application of swap/replace moves to increase
      the total value while maintaining feasibility.
    - Time-budget management: Aggressively cache the best solution found.
    """
    start_time = time.time()
    pieces = instance['pieces']
    n_types = len(pieces)
    
    def get_score(placements):
        return sum(pieces[p[0]-1]['value'] for p in placements)

    def is_valid(placements):
        counts = [0] * n_types
        for p in placements:
            counts[p[0]-1] += 1
        for i in range(n_types):
            if counts[i] < pieces[i]['min'] or counts[i] > pieces[i]['max']:
                return False
        return True

    best_placements = []
    best_score = -1

    # Precompute densities
    densities = []
    for i in range(n_types):
        area = pieces[i]['length'] * pieces[i]['width']
        densities.append((i + 1, pieces[i]['value'] / area if area > 0 else 0))

    # Main Loop
    while time.time() - start_time < time_limit_s * 0.9:
        # 1. Randomized Construction
        # Jitter the density priority to create diverse starting configurations
        jitter = [d[1] * random.uniform(0.5, 1.5) for d in densities]
        order = sorted(range(n_types), key=lambda i: jitter[i], reverse=True)
        
        current_placements = []
        counts = [0] * n_types
        
        # Ensure MIN requirements are met first
        for i in range(n_types):
            while counts[i] < pieces[i]['min']:
                p = tools['try_place_at_corner'](current_placements, i + 1, allow_rotation=True)
                if p:
                    current_placements.append(p)
                    counts[i] += 1
                else: break
        
        # Fill remaining with randomized greedy
        for t_idx in order:
            t = t_idx + 1
            while counts[t-1] < pieces[t-1]['max']:
                p = tools['try_place_at_corner'](current_placements, t, allow_rotation=True)
                if p:
                    current_placements.append(p)
                    counts[t-1] += 1
                else: break
        
        # 2. Local Improvement
        if is_valid(current_placements):
            # Apply swap moves to improve value
            improved = tools['apply_swap_pieces'](current_placements, time_limit_s=min(0.2, (time_limit_s - (time.time() - start_time)) * 0.5))
            
            score = get_score(improved)
            if score > best_score:
                best_score = score
                best_placements = improved

    # Fallback if no valid solution found
    if not best_placements:
        # Use provided greedy tool as a final safety net
        return {'placements': tools['greedy_max_value_density'](allow_rotation=True)}

    return {'placements': best_placements}