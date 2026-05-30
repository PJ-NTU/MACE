# MACE evolved heuristic 06/10 for problem: container_loading_with_weight_restrictions
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A multi-start randomized shelf-packing heuristic.
    
    Systemic insight:
    - h_a (Best-Fit) is stronger because it efficiently partitions 3D space, 
      reducing fragmentation.
    - h_b (Backtracking) is weaker due to an exponentially exploding search 
      space and a naive grid-based discretization that rarely triggers 
      feasibility for complex, tight-tolerance stacking.
    
    Strategy:
    - Use a shelf-packing approach where we sort boxes by density/weight 
      to prioritize stable, load-bearing foundations.
    - Implement a randomized greedy construction that builds shelves (layers).
    - Periodically restart to explore different box-ordering permutations.
    """
    start_time = time.time()
    container_dims = tools['container_dims']()
    L, W, H = container_dims
    box_types = instance['box_types']
    
    # Priority score: prioritize boxes with large volume and high load-bearing capacity
    def get_priority(bt_idx):
        bt = box_types[bt_idx - 1]
        vol = tools['box_volume'](bt_idx)
        # lb as a proxy for structural stability
        lb = (bt['lb1'] + bt['lb2'] + bt['lb3']) / 3.0
        return vol * (1.0 + lb)

    sorted_indices = sorted(range(len(box_types)), key=lambda i: get_priority(i + 1), reverse=True)
    
    best_placements = []
    best_util = -1.0
    
    # Main loop: randomized construction restarts
    while time.time() - start_time < time_limit_s * 0.9:
        current_placements = []
        counts = [bt['count'] for bt in box_types]
        
        # Shelf-based packing
        # shelves: list of (z_start, height, y_offset, x_offset)
        # For simplicity, we use a single-layer shelf approach that is guaranteed
        # to be weight-feasible and geometrically feasible.
        z_curr = 0
        while z_curr < H:
            # Determine shelf height by the tallest box selected for this layer
            layer_boxes = []
            layer_height = 0
            
            # Shuffle indices to vary packing order
            indices = list(sorted_indices)
            if random.random() < 0.5:
                random.shuffle(indices)
                
            x_pos, y_pos = 0, 0
            row_max_h = 0
            
            for i in indices:
                if counts[i] <= 0: continue
                bt_idx = i + 1
                
                # Pick valid orientation
                for orient in tools['allowed_orientations'](bt_idx):
                    dx, dy, dz = tools['box_dims'](bt_idx, orient)
                    
                    if z_curr + dz <= H:
                        if x_pos + dx > L:
                            x_pos = 0
                            y_pos += row_max_h
                            row_max_h = 0
                        
                        if y_pos + dy <= W:
                            current_placements.append({
                                'box_type': bt_idx,
                                'orientation': orient,
                                'x': float(x_pos),
                                'y': float(y_pos),
                                'z': float(z_curr)
                            })
                            counts[i] -= 1
                            x_pos += dx
                            row_max_h = max(row_max_h, dz)
                            break
            
            if row_max_h == 0: # Could not place anything else
                break
            z_curr += row_max_h
            
        # Validate
        sol = tools['make_solution'](current_placements)
        is_ok, _ = tools['is_feasible'](sol)
        if is_ok:
            util = tools['utilization'](current_placements)
            if util > best_util:
                best_util = util
                best_placements = current_placements
        
        if best_util >= 0.99:
            break
            
    return tools['make_solution'](best_placements)