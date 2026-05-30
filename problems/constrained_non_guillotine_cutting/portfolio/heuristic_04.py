# MACE evolved heuristic 04/10 for problem: constrained_non_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized constructive heuristic for 2D non-guillotine cutting.
    Combines density-based ordering, forced minimum demand satisfaction,
    and a robust local search refinement loop.
    """
    start_time = time.time()
    pieces = instance['pieces']
    n_types = len(pieces)
    
    # Precompute metadata
    densities = []
    for i, p in enumerate(pieces):
        area = p['length'] * p['width']
        density = p['value'] / area if area > 0 else 0
        densities.append({'id': i + 1, 'density': density, 'val': p['value'], 'min': p['min'], 'max': p['max']})

    best_placements = []
    best_score = -1

    # Time-budgeted search
    while time.time() - start_time < time_limit_s * 0.9:
        # 1. Randomized Greedy Construction
        # Mix of density and random noise to explore packing configurations
        order = sorted(densities, key=lambda x: x['density'] * random.uniform(0.5, 1.5), reverse=True)
        
        current_placements = []
        counts = [0] * n_types
        
        # Phase 1: Satisfy mandatory minimums
        satisfied_min = True
        for p in densities:
            for _ in range(p['min']):
                res = tools['try_place_at_corner'](current_placements, p['id'], allow_rotation=True)
                if res:
                    current_placements.append(res)
                    counts[p['id']-1] += 1
                else:
                    satisfied_min = False
                    break
            if not satisfied_min: break
        
        if not satisfied_min:
            continue
            
        # Phase 2: Fill remaining space with high-density pieces
        for p in order:
            while counts[p['id']-1] < p['max']:
                res = tools['try_place_at_corner'](current_placements, p['id'], allow_rotation=True)
                if res:
                    current_placements.append(res)
                    counts[p['id']-1] += 1
                else:
                    break
        
        # 2. Local Search Refinement
        # Attempt to swap existing pieces for higher-value ones
        refined = tools['apply_swap_pieces'](current_placements, time_limit_s=0.05)
        if refined:
            current_placements = refined
            
        # 3. Evaluate
        is_f, _ = tools['is_feasible_solution']({'placements': current_placements})
        if is_f:
            score = sum(pieces[p[0]-1]['value'] for p in current_placements)
            if score > best_score:
                best_score = score
                best_placements = list(current_placements)
                
    # Fallback to greedy if no random iteration found a solution
    if not best_placements:
        best_placements = tools['greedy_max_value_density'](allow_rotation=True)
        # Verify min constraints for fallback
        counts = tools['used_count_per_type'](best_placements)
        for i, p in enumerate(pieces):
            if counts[i] < p['min']:
                return {'placements': []}

    return {'placements': best_placements}