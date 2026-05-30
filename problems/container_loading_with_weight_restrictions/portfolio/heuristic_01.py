# MACE evolved heuristic 01/10 for problem: container_loading_with_weight_restrictions
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Best-Fit Decreasing' (BFD) 3D bin packing heuristic with randomized 
    multi-start search. Unlike the portfolio's column-stacked or shelf-based 
    approaches, this treats the space as a 3D grid of 'empty regions' (the 
    'Guillotine' or 'MaxRects' style approach) to pack boxes, rather than 
    forcing a shelf/column structure. 
    
    This approach explores the packing space by partitioning the container 
    into free rectangular sub-volumes and attempting to fill them with 
    the most 'difficult-to-fit' boxes first.
    """
    start_time = time.time()
    container_dims = tools['container_dims']()
    box_types = instance['box_types']
    
    # Track available boxes
    counts = [bt['count'] for bt in box_types]
    
    # A free space is represented as (x, y, z, dx, dy, dz)
    free_spaces = [ (0, 0, 0, container_dims[0], container_dims[1], container_dims[2]) ]
    placements = []
    
    def get_score(bt_idx):
        # Heuristic: prioritize boxes with large volume and high load-bearing capacity
        vol = tools['box_volume'](bt_idx)
        # Average lb
        bt = box_types[bt_idx - 1]
        lb = (bt['lb1'] + bt['lb2'] + bt['lb3']) / 3.0
        return vol * (1.0 + lb)

    # Sort indices by heuristic score
    indices = list(range(len(box_types)))
    indices.sort(key=lambda i: get_score(i + 1), reverse=True)
    
    # Multi-start search
    best_placements = []
    best_util = -1.0
    
    while time.time() - start_time < time_limit_s * 0.9:
        current_placements = []
        current_counts = list(counts)
        # Use a list of free spaces; we manage them via a simple greedy split
        spaces = [ (0, 0, 0, container_dims[0], container_dims[1], container_dims[2]) ]
        
        # Shuffle indices to allow diversity
        random.shuffle(indices)
        
        for i in indices:
            if current_counts[i] <= 0: continue
            bt_idx = i + 1
            
            # Find a space that fits the box
            for s_idx, space in enumerate(spaces):
                sx, sy, sz, sdx, sdy, sdz = space
                
                # Try allowed orientations
                for orient in tools['allowed_orientations'](bt_idx):
                    dx, dy, dz = tools['box_dims'](bt_idx, orient)
                    
                    if dx <= sdx and dy <= sdy and dz <= sdz:
                        # Place box
                        current_placements.append({
                            'box_type': bt_idx,
                            'orientation': orient,
                            'x': sx, 'y': sy, 'z': sz
                        })
                        current_counts[i] -= 1
                        
                        # Split space (basic guillotine-like cut)
                        spaces.pop(s_idx)
                        # Split into 3 new free spaces
                        if sdx - dx > 0: spaces.append((sx + dx, sy, sz, sdx - dx, dy, dz))
                        if sdy - dy > 0: spaces.append((sx, sy + dy, sz, sdx, sdy - dy, dz))
                        if sdz - dz > 0: spaces.append((sx, sy, sz + dz, sdx, sdy, sdz - dz))
                        
                        break # Successfully placed
                else: continue
                break
        
        # Evaluate feasibility and quality
        sol = tools['make_solution'](current_placements)
        is_ok, _ = tools['is_feasible'](sol)
        if is_ok:
            util = tools['utilization'](current_placements)
            if util > best_util:
                best_util = util
                best_placements = current_placements
        
        if best_util > 0.98: break
        
    return tools['make_solution'](best_placements)