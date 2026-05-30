# MACE evolved heuristic 06/10 for problem: constrained_non_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher heuristic for Constrained Non-Guillotine Cutting.
    
    Hypothesis:
    - If the total minimum demand area is high relative to stock area (high load),
      prioritize strict fulfillment of minimums first (B-style).
    - If the pieces are highly heterogeneous in value/area but total demand is low,
      prioritize high-density packing with randomized exploration (A-style).
    """
    start_time = time.time()
    pieces = instance['pieces']
    stock_l, stock_w = tools['stock_dims']()
    stock_area = stock_l * stock_w
    
    # Feature extraction
    min_area = sum(p['min'] * p['length'] * p['width'] for p in pieces)
    load_ratio = min_area / stock_area if stock_area > 0 else 1.0
    
    # Regime selection
    # If load_ratio > 0.6, the problem is constrained by the required pieces (B-style)
    # Otherwise, it's a knapsack-like packing problem (A-style)
    use_b_style = load_ratio > 0.6
    
    n_types = len(pieces)
    type_indices = list(range(1, n_types + 1))
    
    def get_density(t):
        p = pieces[t-1]
        area = p['length'] * p['width']
        return p['value'] / area if area > 0 else 0

    best_placements = []
    best_score = -1.0
    
    while (time.time() - start_time) < (time_limit_s * 0.85):
        current_placements = []
        counts = [0] * n_types
        
        if use_b_style:
            # B-style: Mandatory min fill first
            for i in range(n_types):
                while counts[i] < pieces[i]['min']:
                    p = tools['try_place_at_corner'](current_placements, i + 1, allow_rotation=True)
                    if p:
                        current_placements.append(p)
                        counts[i] += 1
                    else: break
            # Then opportunistic fill
            order = sorted(type_indices, key=get_density, reverse=True)
        else:
            # A-style: Randomized density exploration
            order = sorted(type_indices, key=lambda t: get_density(t) * random.uniform(0.7, 1.3), reverse=True)
            
        for t in order:
            while counts[t-1] < pieces[t-1]['max']:
                p = tools['try_place_at_corner'](current_placements, t, allow_rotation=True)
                if p:
                    current_placements.append(p)
                    counts[t-1] += 1
                else: break
        
        # Verify feasibility
        if all(counts[i] >= pieces[i]['min'] for i in range(n_types)):
            score = sum(pieces[p[0]-1]['value'] for p in current_placements)
            if score > best_score:
                best_score = score
                best_placements = list(current_placements)
                
    # Final refinement
    if not best_placements:
        # Fallback to BFD
        best_placements = tools['bottom_left_pack_BFD'](allow_rotation=True)
        
    refined = tools['apply_swap_pieces'](best_placements, time_limit_s=max(0.05, time_limit_s * 0.05))
    
    # Final safety check
    is_f, _ = tools['is_feasible_solution']({'placements': refined})
    if is_f:
        return {'placements': refined}
    else:
        # Return last known good or empty
        return {'placements': best_placements if best_placements else []}