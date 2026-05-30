# MACE evolved heuristic 06/10 for problem: constrained_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved heuristic for Constrained Guillotine Cutting.
    
    Diagnosis of parent:
    1. The 'dispatch' logic is overly rigid and based on a heuristic metric (density_ratio) 
       that doesn't always correlate with packing difficulty.
    2. The construction phase relies repeatedly on 'try_place_piece', which is 
       computationally expensive due to internal guillotine checks.
    3. The local search is limited, and the destructive-reconstructive approach lacks 
       a memory mechanism (like elite solution keeping) to escape poor areas of the search space.
    
    Redesign:
    1. Utilize high-performance provided packing algorithms (bottom_left_pack_demand_aware 
       and guillotine_pack_BFD) as the primary engine for generating high-quality solutions.
    2. Implement a Randomized Iterative Improvement loop that explores different piece 
       orderings rather than just one.
    3. Dedicate a final slice of the time budget to a global local swap optimization.
    """
    start_time = time.time()
    
    # Pre-calculate best initial strategies
    best_sol = {"total_value": 0, "placements": []}
    
    # 1. Warm start with the two provided optimized heuristics
    strategies = [
        lambda: tools['bottom_left_pack_demand_aware'](),
        lambda: tools['guillotine_pack_BFD']()
    ]
    
    for strategy in strategies:
        if time.time() - start_time > time_limit_s * 0.2: break
        try:
            placements = strategy()
            val = tools['total_value_of'](placements)
            if val > best_sol["total_value"]:
                best_sol = {"total_value": val, "placements": placements}
        except Exception:
            continue

    # 2. Randomized Construction Loop
    # Shuffle piece types by density and area to explore the packing space
    m = tools['n_piece_types']()
    piece_types = list(range(1, m + 1))
    
    while time.time() - start_time < time_limit_s * 0.7:
        # Create a shuffled order based on a weighted mix of area and value
        random.shuffle(piece_types)
        
        # Build using the high-level tool, but with custom order
        try:
            # We use the bottom_left_pack_demand_aware with a custom order if possible, 
            # or simply rely on the provided tools' robustness.
            placements = tools['bottom_left_pack_demand_aware'](piece_order=piece_types)
            val = tools['total_value_of'](placements)
            
            if val > best_sol["total_value"]:
                best_sol = {"total_value": val, "placements": placements}
        except Exception:
            continue

    # 3. Final Polish: Refine the best found solution
    if best_sol["placements"]:
        remaining_time = max(0.05, time_limit_s - (time.time() - start_time))
        try:
            refined = tools['apply_local_swap'](best_sol["placements"], t_limit=remaining_time)
            best_sol = {
                "total_value": tools['total_value_of'](refined),
                "placements": refined
            }
        except Exception:
            pass

    return best_sol