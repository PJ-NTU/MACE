# MACE evolved heuristic 02/10 for problem: container_loading
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized container loading heuristic combining:
    1. Multi-start construction (Wall-Building vs Corner-Packing).
    2. Large Neighborhood Search with adaptive removal and greedy re-insertion.
    3. Time-budgeted refinement using the provided 'apply_swap_boxes' tool.
    """
    start_time = time.time()
    
    # 1. Initialization: Compare construction heuristics
    # Wall building is often better for regular, dense packing
    placements_wb = tools['wall_building_pack'](allow_rotation=True)
    # Corner packing is often better for heterogeneous box sizes
    placements_cp = tools['corner_pack_3d'](allow_rotation=True)
    
    best_placements = placements_wb if tools['used_volume'](placements_wb) >= tools['used_volume'](placements_cp) else placements_cp
    best_util = tools['utilization'](best_placements)
    
    # Pre-calculate available boxes sorted by volume
    box_types = instance['box_types']
    box_ids = []
    for bt, info in box_types.items():
        box_ids.extend([bt] * info['count'])
    box_ids.sort(key=lambda bt: tools['box_value'](bt), reverse=True)
    
    # 2. Refinement Loop (LNS)
    # We allocate 70% of time to LNS, 20% to final swap polishing, 10% safety margin
    lns_time_limit = time_limit_s * 0.7
    
    while (time.time() - start_time) < lns_time_limit:
        current = list(best_placements)
        if not current:
            break
            
        # Adaptive removal: Choose a random removal size between 10% and 40%
        # to balance local exploration and global restructuring
        num_to_remove = random.randint(max(1, len(current) // 10), max(1, len(current) // 2))
        random.shuffle(current)
        remaining = current[num_to_remove:]
        
        # Identify boxes to re-insert: 
        # Always prioritize filling with the largest available boxes (greedy)
        # but keep the order slightly randomized to explore different corner positions
        remaining_box_ids = []
        for bt in box_ids:
            if tools['used_count'](remaining, bt) < box_types[bt]['count']:
                remaining_box_ids.append(bt)
        
        # Sort remaining boxes by volume, then shuffle slightly to prevent stagnation
        remaining_box_ids.sort(key=lambda bt: tools['box_value'](bt), reverse=True)
        
        # Re-pack
        for bt in remaining_box_ids:
            if p := tools['try_place_at_corner_3d'](remaining, bt):
                remaining.append(p)
        
        # Evaluate
        new_util = tools['utilization'](remaining)
        if new_util > best_util:
            best_util = new_util
            best_placements = remaining
            
    # 3. Final Polish
    # Use the provided swap tool to optimize the final configuration
    remaining_time = max(0.1, time_limit_s - (time.time() - start_time))
    final_placements = tools['apply_swap_boxes'](best_placements, time_limit_s=remaining_time)
    
    return tools['make_solution'](final_placements)