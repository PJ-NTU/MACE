# MACE evolved heuristic 04/10 for problem: container_loading_with_weight_restrictions
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Modified BFD heuristic: Prioritizes box selection by volume density
    and improves the packing strategy by sorting free spaces by Z-coordinate
    (bottom-up) to favor floor-first stability, which helps satisfy load-bearing
    constraints more consistently than random space picking.
    """
    start_time = time.time()
    container_dims = tools['container_dims']()
    box_types = instance['box_types']
    
    counts = [bt['count'] for bt in box_types]
    
    def get_score(bt_idx):
        # Heuristic: prioritize boxes with large volume and high load-bearing capacity
        vol = tools['box_volume'](bt_idx)
        bt = box_types[bt_idx - 1]
        lb = (bt['lb1'] + bt['lb2'] + bt['lb3']) / 3.0
        # Incorporate density: volume per unit weight
        weight = tools['box_weight'](bt_idx)
        density = vol / (weight + 1e-6)
        return vol * (1.0 + lb) * math.sqrt(density)

    indices = list(range(len(box_types)))
    indices.sort(key=lambda i: get_score(i + 1), reverse=True)
    
    best_placements = []
    best_util = -1.0
    
    while time.time() - start_time < time_limit_s * 0.9:
        current_placements = []
        current_counts = list(counts)
        # Spaces: (x, y, z, dx, dy, dz)
        spaces = [ (0, 0, 0, container_dims[0], container_dims[1], container_dims[2]) ]
        
        # Shuffle indices slightly for exploration
        temp_indices = [i for i in indices if random.random() > 0.1]
        
        for i in temp_indices:
            if current_counts[i] <= 0: continue
            bt_idx = i + 1
            
            # Sort spaces: prioritize lower Z, then lower Y, then lower X (bottom-left-front)
            spaces.sort(key=lambda s: (s[2], s[1], s[0]))
            
            for s_idx, space in enumerate(spaces):
                sx, sy, sz, sdx, sdy, sdz = space
                
                # Try allowed orientations
                for orient in tools['allowed_orientations'](bt_idx):
                    dx, dy, dz = tools['box_dims'](bt_idx, orient)
                    
                    if dx <= sdx and dy <= sdy and dz <= sdz:
                        current_placements.append({
                            'box_type': bt_idx,
                            'orientation': orient,
                            'x': sx, 'y': sy, 'z': sz
                        })
                        current_counts[i] -= 1
                        
                        spaces.pop(s_idx)
                        # Split into new free spaces
                        if sdx - dx > 0: spaces.append((sx + dx, sy, sz, sdx - dx, dy, dz))
                        if sdy - dy > 0: spaces.append((sx, sy + dy, sz, dx, sdy - dy, dz))
                        if sdz - dz > 0: spaces.append((sx, sy, sz + dz, dx, dy, sdz - dz))
                        
                        break 
                else: continue
                break
        
        sol = tools['make_solution'](current_placements)
        is_ok, _ = tools['is_feasible'](sol)
        if is_ok:
            util = tools['utilization'](current_placements)
            if util > best_util:
                best_util = util
                best_placements = current_placements
        
        if best_util > 0.98: break
        
    return tools['make_solution'](best_placements)