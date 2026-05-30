# MACE evolved heuristic 07/10 for problem: constrained_non_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized heuristic using multi-start greedy search with a focus on 
    adaptive density-based packing and robust constraint satisfaction.
    
    Modification: Enhanced the local refinement phase by iterative greedy 
    filling after the initial swap-improvement to densify the packing.
    """
    start_time = time.time()
    pieces = instance['pieces']
    num_types = len(pieces)
    
    # Precompute metadata
    piece_data = []
    for i in range(num_types):
        p = pieces[i]
        area = p['length'] * p['width']
        density = p['value'] / area if area > 0 else 0
        piece_data.append({
            'id': i + 1,
            'density': density,
            'min': p['min'],
            'max': p['max'],
            'value': p['value']
        })

    best_placements = []
    best_score = -1

    # Heuristic loop
    while time.time() - start_time < time_limit_s * 0.9:
        current_placements = []
        counts = [0] * num_types
        
        # Strategy selection: 
        # 0: Min-requirement first (Constraint-Heavy)
        # 1: Pure density-based (Value-Heavy)
        strategy = 0 if random.random() < 0.3 else 1
        
        if strategy == 0:
            # Force mandatory
            mandatory = [p for p in piece_data if p['min'] > 0]
            random.shuffle(mandatory)
            for p_info in mandatory:
                while counts[p_info['id'] - 1] < p_info['min']:
                    placement = tools['try_place_at_corner'](current_placements, p_info['id'], allow_rotation=True)
                    if placement:
                        current_placements.append(placement)
                        counts[p_info['id'] - 1] += 1
                    else: break
        
        # Fill remaining
        # Use a soft-sorting bias: sort by density but add noise
        sorted_types = sorted(piece_data, key=lambda x: x['density'] * random.uniform(0.7, 1.3), reverse=True)
        
        for p_info in sorted_types:
            while counts[p_info['id'] - 1] < p_info['max']:
                placement = tools['try_place_at_corner'](current_placements, p_info['id'], allow_rotation=True)
                if placement:
                    current_placements.append(placement)
                    counts[p_info['id'] - 1] += 1
                else:
                    break
        
        # Local refinement: improved by performing additional greedy filling after swaps
        improved = tools['apply_swap_pieces'](current_placements, time_limit_s=0.02)
        
        # Post-swap greedy densification: try to add any remaining pieces that fit
        for p_info in sorted(piece_data, key=lambda x: x['density'], reverse=True):
            current_counts = tools['used_count_per_type'](improved)
            if current_counts[p_info['id'] - 1] < p_info['max']:
                placement = tools['try_place_at_corner'](improved, p_info['id'], allow_rotation=True)
                if placement:
                    improved.append(placement)
        
        # Final validity check
        is_valid, _ = tools['is_feasible_solution']({'placements': improved})
        if is_valid:
            current_score = sum(pieces[p[0] - 1]['value'] for p in improved)
            if current_score > best_score:
                best_score = current_score
                best_placements = list(improved)
    
    # If no valid solution found, use the standard library-based fallback
    if not best_placements:
        fallback = tools['greedy_max_value_density'](allow_rotation=True)
        return {'placements': fallback}
        
    return {'placements': best_placements}