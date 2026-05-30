# MACE evolved heuristic 08/10 for problem: container_loading_with_weight_restrictions
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved shelf-packing heuristic with a refined selection strategy:
    instead of just shuffling, we implement a 'weighted roulette' selection 
    that biases towards higher-density boxes while maintaining exploration.
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

    priorities = [get_priority(i + 1) for i in range(len(box_types))]
    
    best_placements = []
    best_util = -1.0
    
    # Main loop: randomized construction restarts
    while time.time() - start_time < time_limit_s * 0.9:
        current_placements = []
        counts = [bt['count'] for bt in box_types]
        
        z_curr = 0
        while z_curr < H:
            layer_height = 0
            
            # Weighted Roulette: pick boxes based on priority score
            indices = list(range(len(box_types)))
            weights = [priorities[i] if counts[i] > 0 else 0 for i in indices]
            if sum(weights) == 0:
                break
            
            x_pos, y_pos = 0, 0
            row_max_h = 0
            
            # Try to fill a shelf
            for _ in range(len(box_types)):
                # Weighted selection
                choice = random.choices(indices, weights=weights, k=1)[0]
                if counts[choice] <= 0: continue
                
                bt_idx = choice + 1
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
                            counts[choice] -= 1
                            weights[choice] = priorities[choice] if counts[choice] > 0 else 0
                            x_pos += dx
                            row_max_h = max(row_max_h, dz)
                            break
            
            if row_max_h == 0:
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