# MACE evolved heuristic 04/10 for problem: container_loading
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized container loading heuristic with a weighted mutation strategy:
    The construction phase now uses a 'Biased-Randomized' approach for box orders
    to explore the search space more effectively than pure deterministic sorting.
    """
    start_time = time.time()
    
    # 1. Initialization: Compare construction heuristics
    placements_wb = tools['wall_building_pack'](allow_rotation=True)
    placements_cp = tools['corner_pack_3d'](allow_rotation=True)
    
    best_placements = placements_wb if tools['used_volume'](placements_wb) >= tools['used_volume'](placements_cp) else placements_cp
    best_util = tools['utilization'](best_placements)
    
    # Pre-calculate available boxes
    box_types = instance['box_types']
    box_ids = []
    for bt, info in box_types.items():
        box_ids.extend([bt] * info['count'])
    
    # Sort by volume for the base greedy strategy
    box_ids.sort(key=lambda bt: tools['box_value'](bt), reverse=True)
    
    # 2. Refinement Loop (LNS with Weighted Mutation)
    lns_time_limit = time_limit_s * 0.7
    
    while (time.time() - start_time) < lns_time_limit:
        current = list(best_placements)
        if not current:
            break
            
        num_to_remove = random.randint(max(1, len(current) // 10), max(1, len(current) // 2))
        random.shuffle(current)
        remaining = current[num_to_remove:]
        
        # Weighted Mutation: instead of strictly largest-first, introduce a 
        # probability-based shuffle for the insertion order to escape local optima
        available_list = []
        for bt in box_ids:
            if tools['used_count'](remaining, bt) < box_types[bt]['count']:
                available_list.append(bt)
        
        # Apply weighted mutation: 20% of the time, randomize insertion order,
        # otherwise maintain volume-based priority
        if random.random() > 0.2:
            available_list.sort(key=lambda bt: tools['box_value'](bt), reverse=True)
        else:
            random.shuffle(available_list)
        
        # Re-pack
        for bt in available_list:
            if p := tools['try_place_at_corner_3d'](remaining, bt):
                remaining.append(p)
        
        # Evaluate
        new_util = tools['utilization'](remaining)
        if new_util > best_util:
            best_util = new_util
            best_placements = remaining
            
    # 3. Final Polish
    remaining_time = max(0.1, time_limit_s - (time.time() - start_time))
    final_placements = tools['apply_swap_boxes'](best_placements, time_limit_s=remaining_time)
    
    return tools['make_solution'](final_placements)