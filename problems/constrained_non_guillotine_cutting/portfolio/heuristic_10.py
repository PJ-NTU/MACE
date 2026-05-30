# MACE evolved heuristic 10/10 for problem: constrained_non_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized heuristic for 2D non-guillotine cutting.
    
    Design Philosophy:
    1. Robust Construction: Prioritizes mandatory pieces strictly, then uses 
       stochastic density-based packing for remaining space to maximize value.
    2. Adaptive Search: Uses a multi-start framework with noise-weighted density 
       ordering to explore diverse packing layouts.
    3. Time-Aware Refinement: Applies local swap improvements only when time permits 
       to maximize the number of full packing iterations within the budget.
    4. Guarded Fallback: Ensures a feasible solution is always returned by 
       prioritizing the best valid result found or a default greedy packing.
    """
    start_time = time.time()
    pieces = instance['pieces']
    n_types = len(pieces)
    
    # Precompute metadata for efficient access
    piece_meta = []
    for i, p in enumerate(pieces):
        area = p['length'] * p['width']
        density = p['value'] / area if area > 0 else 0
        piece_meta.append({
            'id': i + 1,
            'density': density,
            'min': p['min'],
            'max': p['max'],
            'value': p['value']
        })

    best_placements = []
    best_score = -1

    # Main search loop: Keep iterating until time limit is reached
    while time.time() - start_time < time_limit_s * 0.95:
        current_placements = []
        counts = [0] * n_types
        
        # 1. Mandatory Satisfaction: Always prioritize minimum requirements
        # Shuffle IDs to allow different packing sequences for mandatory pieces
        mandatory_ids = [p['id'] for p in piece_meta if p['min'] > 0]
        random.shuffle(mandatory_ids)
        
        success = True
        for m_id in mandatory_ids:
            while counts[m_id - 1] < pieces[m_id - 1]['min']:
                placement = tools['try_place_at_corner'](current_placements, m_id, allow_rotation=True)
                if placement:
                    current_placements.append(placement)
                    counts[m_id - 1] += 1
                else:
                    success = False
                    break
            if not success: break
        
        if not success:
            continue
            
        # 2. Opportunistic Filling: Fill remaining slots with high-density pieces
        # Use stochastic ranking to explore the search space
        sorted_types = sorted(
            piece_meta, 
            key=lambda x: x['density'] * random.uniform(0.8, 1.2), 
            reverse=True
        )
        
        for p_info in sorted_types:
            while counts[p_info['id'] - 1] < p_info['max']:
                placement = tools['try_place_at_corner'](current_placements, p_info['id'], allow_rotation=True)
                if placement:
                    current_placements.append(placement)
                    counts[p_info['id'] - 1] += 1
                else:
                    break
        
        # 3. Local Refinement: Attempt to improve value via swaps
        # Only run if we have enough time remaining
        remaining_time = time_limit_s - (time.time() - start_time)
        if remaining_time > 0.05:
            current_placements = tools['apply_swap_pieces'](current_placements, time_limit_s=min(0.05, remaining_time))
            
        # 4. Global Feasibility Check
        is_valid, _ = tools['is_feasible_solution']({'placements': current_placements})
        if is_valid:
            current_score = sum(pieces[p[0] - 1]['value'] for p in current_placements)
            if current_score > best_score:
                best_score = current_score
                best_placements = list(current_placements)
                
    # Fallback: If no valid solution found (or potentially zero-value valid), 
    # use the library's robust greedy heuristic.
    if not best_placements:
        fallback = tools['greedy_max_value_density'](allow_rotation=True)
        # Final safety check
        is_f, _ = tools['is_feasible_solution']({'placements': fallback})
        if is_f:
            return {'placements': fallback}
        return {'placements': []}

    return {'placements': best_placements}