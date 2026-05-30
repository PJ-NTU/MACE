# MACE evolved heuristic 03/10 for problem: container_loading
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized hybrid heuristic that combines deterministic construction 
    with a focused iterative improvement loop.
    
    Strategy:
    1. Initial Construction: Generate a baseline using both Wall-Building and 
       Corner-Packing to select the best starting point.
    2. Adaptive Local Search: Use a LNS-inspired (Large Neighborhood Search) 
       perturbation strategy that prioritizes removing boxes that block 
       future placements, then re-fills using a greedy approach.
    3. Time-Awareness: Dynamically scale the number of iterations based 
       on the remaining time budget.
    """
    start_time = time.time()
    container = instance['container']
    box_types = instance['box_types']
    
    # Pre-calculate all boxes to be placed
    all_boxes = []
    for bt, info in box_types.items():
        all_boxes.extend([bt] * info['count'])
    
    # Sort boxes by volume descending for effective greedy packing
    def get_vol(bt):
        dims = box_types[bt]['dims']
        return dims[0] * dims[1] * dims[2]
    
    sorted_boxes = sorted(all_boxes, key=get_vol, reverse=True)
    
    # 1. Initial baseline
    best_placements = tools['corner_pack_3d'](box_order=sorted_boxes, allow_rotation=True)
    wall_placements = tools['wall_building_pack'](box_order=sorted_boxes, allow_rotation=True)
    
    if tools['utilization'](wall_placements) > tools['utilization'](best_placements):
        best_placements = wall_placements
        
    best_util = tools['utilization'](best_placements)
    
    # 2. Iterative Improvement
    # We use a controlled destruction/construction loop
    while (time.time() - start_time) < (time_limit_s * 0.9):
        # Destruct: Remove random 20%
        if not best_placements:
            break
            
        current_placements = list(best_placements)
        random.shuffle(current_placements)
        num_remove = max(1, len(current_placements) // 5)
        del current_placements[:num_remove]
        
        # Construct: Shuffle remaining boxes and attempt to fill gaps
        # We prioritize adding boxes that weren't in the current set
        placed_types = [p['box_type'] for p in current_placements]
        remaining = []
        for bt in all_boxes:
            if placed_types.count(bt) < box_types[bt]['count']:
                remaining.append(bt)
        
        random.shuffle(remaining)
        
        # Greedy fill
        for bt in remaining:
            placement = tools['try_place_at_corner_3d'](current_placements, bt)
            if placement:
                current_placements.append(placement)
        
        # Evaluate
        current_util = tools['utilization'](current_placements)
        if current_util > best_util:
            best_util = current_util
            best_placements = list(current_placements)
            
        # Early exit if we hit near-optimal
        if best_util >= 0.99:
            break
            
    return tools['make_solution'](best_placements)