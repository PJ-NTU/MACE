# MACE evolved heuristic 02/10 for problem: container_loading_with_weight_restrictions
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-based solver for Container Loading.
    
    Hypothesis: 
    - The column-stacked approach (A) is superior for instances with heavy, 
      high-load-bearing boxes where vertical stability is the bottleneck.
    - The guillotine/space-partitioning approach (B) is superior for instances 
      with many small, light, or heterogeneous items where maximizing 
      3D volume utilization requires complex spatial fragmentation.
    
    Decision Criterion:
    Calculate the 'Density Potential' of the box types. If the average 
    load-bearing capacity-to-weight ratio is low, vertical stacking (A) 
    is risky and likely to fail; we default to the more flexible space-partitioning (B).
    If the ratio is high, we prefer the structure of (A).
    """
    start_time = time.time()
    container = tools['container_dims']()
    box_types = instance['box_types']
    
    # Calculate load-bearing feasibility score
    # A high score means boxes can support many others (Structure-friendly)
    total_lb_units = 0
    for bt in box_types:
        avg_lb = (bt['lb1'] + bt['lb2'] + bt['lb3']) / 3.0
        # Weight-to-capacity ratio: how heavy is the box vs how much it can hold
        if bt['weight'] > 0:
            total_lb_units += (avg_lb / bt['weight'])
    
    avg_lb_score = total_lb_units / max(1, len(box_types))
    
    # Dispatch: If avg_lb_score is high, use column stacking (A).
    # If low, use guillotine-style partitioning (B).
    if avg_lb_score > 0.5:
        # Implementation of A (Column Stacked)
        best_placements = tools['solve_column_stacked']()
        best_util = tools['utilization'](best_placements)
        indices = list(range(len(box_types)))
        
        while time.time() - start_time < time_limit_s * 0.9:
            random.shuffle(indices)
            candidate = []
            counts = [bt['count'] for bt in box_types]
            curr_x, curr_y, row_max_dy = 0.0, 0.0, 0.0
            
            for idx in indices:
                for orient in tools['allowed_orientations'](idx + 1):
                    dx, dy, dz = tools['box_dims'](idx + 1, orient)
                    max_h = tools['max_column_height'](idx + 1, orient)
                    if max_h <= 0 or counts[idx] < max_h: continue
                    
                    if curr_x + dx > container[0]:
                        curr_x = 0
                        curr_y += row_max_dy
                        row_max_dy = 0.0
                    if curr_y + dy > container[1]: break
                    
                    for z in range(max_h):
                        candidate.append({'box_type': idx + 1, 'orientation': orient, 'x': curr_x, 'y': curr_y, 'z': z * dz})
                    counts[idx] -= max_h
                    curr_x += dx
                    row_max_dy = max(row_max_dy, dy)
                    break
            
            sol = tools['make_solution'](candidate)
            if tools['is_feasible'](sol)[0]:
                util = tools['utilization'](candidate)
                if util > best_util:
                    best_util, best_placements = util, candidate
        return tools['make_solution'](best_placements)
    
    else:
        # Implementation of B (Guillotine/Space Partitioning)
        best_placements = []
        best_util = -1.0
        indices = sorted(range(len(box_types)), key=lambda i: tools['box_volume'](i + 1), reverse=True)
        
        while time.time() - start_time < time_limit_s * 0.9:
            current = []
            counts = [bt['count'] for bt in box_types]
            spaces = [(0, 0, 0, container[0], container[1], container[2])]
            random.shuffle(indices)
            
            for i in indices:
                for s_idx, (sx, sy, sz, sdx, sdy, sdz) in enumerate(spaces):
                    for orient in tools['allowed_orientations'](i + 1):
                        dx, dy, dz = tools['box_dims'](i + 1, orient)
                        if dx <= sdx and dy <= sdy and dz <= sdz and counts[i] > 0:
                            current.append({'box_type': i + 1, 'orientation': orient, 'x': sx, 'y': sy, 'z': sz})
                            counts[i] -= 1
                            spaces.pop(s_idx)
                            if sdx - dx > 0: spaces.append((sx + dx, sy, sz, sdx - dx, dy, dz))
                            if sdy - dy > 0: spaces.append((sx, sy + dy, sz, sdx, sdy - dy, dz))
                            if sdz - dz > 0: spaces.append((sx, sy, sz + dz, sdx, sdy, sdz - dz))
                            break
                    else: continue
                    break
            
            sol = tools['make_solution'](current)
            if tools['is_feasible'](sol)[0]:
                util = tools['utilization'](current)
                if util > best_util:
                    best_util, best_placements = util, current
        return tools['make_solution'](best_placements)