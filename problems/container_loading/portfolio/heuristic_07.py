# MACE evolved heuristic 07/10 for problem: container_loading
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized Container Loading:
    The parent heuristic suffered from slow, ineffective LNS and redundant 
    construction logic. This version uses a multi-start greedy approach with 
    a focus on high-density packing, followed by a time-aware perturbation 
    phase.
    """
    start_time = time.time()
    
    # 1. Diversified Construction
    # We generate multiple initial candidates using different strategies.
    # Wall-building is robust for structured loads, while corner_pack_3d is 
    # better for irregular mixes. We randomize the order to explore the space.
    box_types = instance['box_types']
    box_ids = list(box_types.keys())
    
    candidates = []
    
    # Strategy A: Volume-weighted greedy
    candidates.append(tools['wall_building_pack'](allow_rotation=True))
    
    # Strategy B: Randomized greedy corner packing
    for _ in range(5):
        random.shuffle(box_ids)
        candidates.append(tools['corner_pack_3d'](box_order=box_ids, allow_rotation=True))
        if time.time() - start_time > time_limit_s * 0.2:
            break
            
    # Select best starting point
    best_placements = max(candidates, key=lambda p: tools['used_volume'](p))
    
    # 2. Iterative Improvement (Hill Climbing with Perturbation)
    # Instead of full SA, we perform targeted removals and re-insertions 
    # to maximize volume utilization.
    
    iteration = 0
    while (time.time() - start_time) < (time_limit_s * 0.9):
        iteration += 1
        
        # Perturbation: Remove small subset to free up space
        if not best_placements:
            break
            
        removal_size = max(1, len(best_placements) // 4)
        indices = random.sample(range(len(best_placements)), removal_size)
        indices.sort(reverse=True)
        
        current_placements = list(best_placements)
        for idx in indices:
            current_placements.pop(idx)
            
        # Refill: Try to insert remaining boxes
        # We prioritize larger boxes to fill the gaps created
        remaining_boxes = []
        for bt in box_ids:
            count_in = tools['used_count'](current_placements, bt)
            for _ in range(box_types[bt]['count'] - count_in):
                remaining_boxes.append(bt)
        
        remaining_boxes.sort(key=lambda bt: tools['box_value'](bt), reverse=True)
        
        for bt in remaining_boxes:
            if p := tools['try_place_at_corner_3d'](current_placements, bt):
                current_placements.append(p)
                
        # Acceptance: Keep if better
        if tools['used_volume'](current_placements) > tools['used_volume'](best_placements):
            best_placements = current_placements
            
    # 3. Final Polish
    # Use the reliable swap tool to perform local container-space optimizations
    remaining = max(0.1, time_limit_s - (time.time() - start_time))
    final_placements = tools['apply_swap_boxes'](best_placements, time_limit_s=remaining)
    
    return tools['make_solution'](final_placements)