# MACE evolved heuristic 03/10 for problem: container_loading_with_weight_restrictions
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Modified BFD heuristic: improved the space partitioning strategy.
    Instead of simple guillotine cuts which often create narrow, unusable 
    slivers, this version uses a 'Best-Fit' space selection strategy:
    it selects the free space with the smallest residual volume (Best-Fit)
    that can accommodate the box, reducing fragmentation and leaving larger
    contiguous volumes for subsequent boxes.
    """
    start_time = time.time()
    container_dims = tools['container_dims']()
    box_types = instance['box_types']
    
    counts = [bt['count'] for bt in box_types]
    
    def get_score(bt_idx):
        vol = tools['box_volume'](bt_idx)
        bt = box_types[bt_idx - 1]
        lb = (bt['lb1'] + bt['lb2'] + bt['lb3']) / 3.0
        return vol * (1.0 + lb)

    indices = list(range(len(box_types)))
    indices.sort(key=lambda i: get_score(i + 1), reverse=True)
    
    best_placements = []
    best_util = -1.0
    
    while time.time() - start_time < time_limit_s * 0.9:
        current_placements = []
        current_counts = list(counts)
        # Use a list of free spaces; manage via Best-Fit
        spaces = [ (0, 0, 0, container_dims[0], container_dims[1], container_dims[2]) ]
        
        # Partially randomize the order to explore different packing sequences
        shuffled_indices = list(indices)
        if random.random() < 0.3:
            random.shuffle(shuffled_indices)
        
        for i in shuffled_indices:
            if current_counts[i] <= 0: continue
            bt_idx = i + 1
            
            best_space_idx = -1
            best_space_vol = float('inf')
            best_orient = -1
            
            # Find the "tightest" fitting space (Best-Fit)
            for s_idx, space in enumerate(spaces):
                sx, sy, sz, sdx, sdy, sdz = space
                for orient in tools['allowed_orientations'](bt_idx):
                    dx, dy, dz = tools['box_dims'](bt_idx, orient)
                    if dx <= sdx and dy <= sdy and dz <= sdz:
                        vol = sdx * sdy * sdz
                        if vol < best_space_vol:
                            best_space_vol = vol
                            best_space_idx = s_idx
                            best_orient = orient
            
            if best_space_idx != -1:
                sx, sy, sz, sdx, sdy, sdz = spaces.pop(best_space_idx)
                dx, dy, dz = tools['box_dims'](bt_idx, best_orient)
                
                current_placements.append({
                    'box_type': bt_idx,
                    'orientation': best_orient,
                    'x': sx, 'y': sy, 'z': sz
                })
                current_counts[i] -= 1
                
                # Split space into 3 remaining sub-regions
                if sdx - dx > 0: spaces.append((sx + dx, sy, sz, sdx - dx, dy, dz))
                if sdy - dy > 0: spaces.append((sx, sy + dy, sz, sdx, sdy - dy, dz))
                if sdz - dz > 0: spaces.append((sx, sy, sz + dz, sdx, sdy, sdz - dz))
        
        sol = tools['make_solution'](current_placements)
        is_ok, _ = tools['is_feasible'](sol)
        if is_ok:
            util = tools['utilization'](current_placements)
            if util > best_util:
                best_util = util
                best_placements = current_placements
        
        if best_util > 0.98: break
        
    return tools['make_solution'](best_placements)