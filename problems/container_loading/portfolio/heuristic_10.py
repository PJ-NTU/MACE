# MACE evolved heuristic 10/10 for problem: container_loading
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized container loading using a multi-pass approach:
    1. Generate base solutions using both wall-building and corner-packing.
    2. Maintain the best found solution.
    3. Use a time-budgeted hill climbing (apply_swap_boxes) to refine the best solution.
    """
    start_time = time.time()
    
    # 1. Generate initial candidate solutions
    # We use both construction heuristics as they excel in different box distributions
    candidates = []
    
    # Attempt wall building
    try:
        sol_wall = tools['wall_building_pack']()
        candidates.append(sol_wall)
    except Exception:
        pass
        
    # Attempt corner packing
    try:
        sol_corner = tools['corner_pack_3d']()
        candidates.append(sol_corner)
    except Exception:
        pass
    
    # 2. Select initial best
    best_placements = []
    best_util = -1.0
    
    for c in candidates:
        util = tools['utilization'](c)
        if util > best_util:
            best_util = util
            best_placements = c
            
    # 3. Refinement Phase (Local Search)
    # We allocate 85% of remaining time to refinement, ensuring we return before deadline
    elapsed = time.time() - start_time
    refinement_limit = max(0.5, (time_limit_s - elapsed) * 0.85)
    
    if best_placements:
        refined = tools['apply_swap_boxes'](best_placements, time_limit_s=refinement_limit)
        if tools['utilization'](refined) > best_util:
            best_placements = refined
            
    return tools['make_solution'](best_placements)