# MACE evolved heuristic 07/10 for problem: container_loading_with_weight_restrictions
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher heuristic:
    - If the container is 'deep/tall' (high Z dimension) or the box count is 
      high relative to volume, it favors stacking strategies (column-based).
    - If the container is 'flat/wide' or box counts are low, it favors 
      Best-Fit space decomposition strategies.
      
    Criteria: Aspect ratio and volume utilization potential.
    """
    start_time = time.time()
    container = instance['container']
    L, W, H = container
    container_vol = L * W * H
    
    # Heuristic feature: Box diversity and Container Shape
    # High H/L ratio suggests stacking is efficient.
    aspect_ratio = H / max(L, W)
    
    # Calculate total potential cargo volume
    total_cargo_vol = sum(tools['box_volume'](i + 1) * instance['box_types'][i]['count'] 
                          for i in range(instance['n']))
    density = total_cargo_vol / container_vol

    # Strategy A: Column Stacking (Robust for stacking constraints)
    def solve_stacking():
        # Using tools provided column_stacked which is structurally safe
        placements = tools['solve_column_stacked']()
        return tools['make_solution'](placements)

    # Strategy B: Best-Fit space decomposition (Robust for floor-filling/fragmentation)
    def solve_best_fit():
        box_types = instance['box_types']
        counts = [bt['count'] for bt in box_types]
        indices = list(range(len(box_types)))
        # Sort by volume * weight/lb potential
        indices.sort(key=lambda i: tools['box_volume'](i + 1) * (1.0 + (sum([box_types[i]['lb1'], box_types[i]['lb2'], box_types[i]['lb3']]) / 3.0)), reverse=True)
        
        best_placements = []
        best_util = -1.0
        
        while time.time() - start_time < time_limit_s * 0.8:
            current_placements = []
            current_counts = list(counts)
            spaces = [(0, 0, 0, L, W, H)]
            
            for i in indices:
                if current_counts[i] <= 0: continue
                bt_idx = i + 1
                best_space_idx = -1
                best_space_vol = float('inf')
                best_orient = -1
                
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
                    current_placements.append({'box_type': bt_idx, 'orientation': best_orient, 'x': sx, 'y': sy, 'z': sz})
                    current_counts[i] -= 1
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
        return tools['make_solution'](best_placements)

    # Decision logic
    if aspect_ratio > 0.8 or density > 1.5:
        return solve_stacking()
    else:
        return solve_best_fit()