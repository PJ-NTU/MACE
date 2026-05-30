# MACE evolved heuristic 09/10 for problem: container_loading
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized container loading heuristic combining:
    1. Multi-strategy initialization (Wall-Building + Corner-Packing).
    2. Adaptive LNS (Large Neighborhood Search) with intelligent destruction.
    3. Randomized greedy repair with volume-first ordering.
    4. Strict time-budget monitoring and final polish.
    """
    start_time = time.time()
    container_vol = float(instance['container'][0] * instance['container'][1] * instance['container'][2])
    box_types = instance['box_types']
    
    # Precompute all available box instances
    all_box_instances = []
    for bt, info in box_types.items():
        all_box_instances.extend([bt] * info['count'])
    
    # Sort for greedy initialization
    all_box_instances.sort(key=lambda bt: tools['box_value'](bt), reverse=True)
    
    # 1. Initialization: Compare Wall-Building and Corner-Packing
    p_wall = tools['wall_building_pack'](box_order=all_box_instances, allow_rotation=True)
    p_corner = tools['corner_pack_3d'](box_order=all_box_instances, allow_rotation=True)
    
    best_placements = p_wall if tools['used_volume'](p_wall) >= tools['used_volume'](p_corner) else p_corner
    best_vol = tools['used_volume'](best_placements)
    
    # 2. Iterative Improvement Phase (LNS)
    # We use a time-budgeted loop to perform destruction and repair
    # The destruction probability is adaptive: range 10-40%
    while (time.time() - start_time) < (time_limit_s * 0.85):
        if not best_placements:
            break
            
        current = list(best_placements)
        # Destruction: Remove a random subset
        num_remove = random.randint(max(1, len(current) // 8), max(1, len(current) // 3))
        random.shuffle(current)
        remaining = current[num_remove:]
        
        # Repair: Greedy Fill with shuffled candidates
        # Re-collect remaining available boxes
        used_counts = {}
        for p in remaining:
            bt = p['box_type']
            used_counts[bt] = used_counts.get(bt, 0) + 1
            
        available = []
        for bt in box_types:
            needed = box_types[bt]['count'] - used_counts.get(bt, 0)
            if needed > 0:
                available.extend([bt] * needed)
        
        random.shuffle(available)
        
        # Attempt to re-fill
        for bt in available:
            if p := tools['try_place_at_corner_3d'](remaining, bt):
                remaining.append(p)
        
        # Acceptance: Only keep if improvement is found
        new_vol = tools['used_volume'](remaining)
        if new_vol > best_vol:
            best_vol = new_vol
            best_placements = remaining
            
    # 3. Final Polish: Use the optimized tool for the remaining time
    remaining_time = max(0.1, time_limit_s - (time.time() - start_time))
    try:
        best_placements = tools['apply_swap_boxes'](best_placements, time_limit_s=remaining_time)
    except Exception:
        pass # Fallback to best_placements if swap fails
        
    return tools['make_solution'](best_placements)