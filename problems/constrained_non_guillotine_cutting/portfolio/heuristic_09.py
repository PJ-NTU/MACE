# MACE evolved heuristic 09/10 for problem: constrained_non_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized heuristic for Constrained Non-Guillotine Cutting.
    
    The strategy employs:
    1. A multi-phase construction ensuring hard constraints (MIN-demand) are met
       before filling the remaining space.
    2. Randomized greedy ordering (GRASP) using density-based priority with 
       stochastic jitter to explore the packing space.
    3. A robust local search refinement loop using apply_swap_pieces.
    4. Adaptive time monitoring to ensure the best feasible solution is returned.
    """
    start_time = time.time()
    pieces = instance['pieces']
    n_types = len(pieces)
    
    # Precompute densities for sorting
    piece_data = []
    for i, p in enumerate(pieces):
        area = p['length'] * p['width']
        density = p['value'] / area if area > 0 else 0
        piece_data.append({
            'id': i + 1, 
            'density': density, 
            'min': p['min'], 
            'max': p['max']
        })

    best_placements = []
    best_score = -1

    # Main optimization loop
    # We allocate 90% of the time to searching, leaving 10% for final cleanup/safety
    while time.time() - start_time < time_limit_s * 0.9:
        # Stochastic ordering: prioritize density but add noise to explore permutations
        # This helps break local optima in the packing sequence
        order = sorted(piece_data, key=lambda x: x['density'] * random.uniform(0.6, 1.4), reverse=True)
        
        current_placements = []
        counts = [0] * n_types
        
        # Phase 1: Force satisfaction of mandatory minimums
        # Using a deterministic order for min-requirements ensures higher feasibility rates
        possible = True
        for p in piece_data:
            for _ in range(p['min']):
                res = tools['try_place_at_corner'](current_placements, p['id'], allow_rotation=True)
                if res:
                    current_placements.append(res)
                    counts[p['id'] - 1] += 1
                else:
                    possible = False
                    break
            if not possible: break
        
        if not possible:
            continue
            
        # Phase 2: Fill remaining space based on randomized density order
        for p in order:
            while counts[p['id'] - 1] < p['max']:
                res = tools['try_place_at_corner'](current_placements, p['id'], allow_rotation=True)
                if res:
                    current_placements.append(res)
                    counts[p['id'] - 1] += 1
                else:
                    break
        
        # Phase 3: Local Search Refinement
        # Apply swap to improve the value of existing placements
        refined = tools['apply_swap_pieces'](current_placements, time_limit_s=min(0.05, time_limit_s * 0.05))
        if refined:
            current_placements = refined
            
        # Phase 4: Evaluate and update global best
        is_f, _ = tools['is_feasible_solution']({'placements': current_placements})
        if is_f:
            score = sum(pieces[p[0] - 1]['value'] for p in current_placements)
            if score > best_score:
                best_score = score
                best_placements = list(current_placements)
                
    # Final check: if no feasible solution found, use the greedy baseline
    if not best_placements:
        fallback = tools['greedy_max_value_density'](allow_rotation=True)
        # Verify min constraints for fallback fallback
        c = tools['used_count_per_type'](fallback)
        if all(c[i] >= pieces[i]['min'] for i in range(n_types)):
            best_placements = fallback
        else:
            return {'placements': []}

    return {'placements': best_placements}