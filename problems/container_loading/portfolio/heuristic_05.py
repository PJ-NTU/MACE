# MACE evolved heuristic 05/10 for problem: container_loading
import time
import random
import itertools

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Packing-Space Decomposition' heuristic.
    
    Departure from portfolio:
    1. Instead of generic LNS or simple swap-boxes, this uses a 'Bin-Splitting' 
       strategy: it recursively partitions the available free space into 
       orthogonal sub-boxes (rectangles) and fits items into the smallest 
       matching sub-box (Best-Fit Space Allocation).
    2. It uses a 'Deterministic Priority Queue' approach for space assignment 
       rather than just corner-point packing.
    3. It prioritizes filling small, awkward volumes first to minimize 
       fragmentation, whereas the portfolio mostly focuses on Large-First 
       greedy packing.
    """
    start_time = time.time()
    container_dims = tools['container_dims']()
    box_types = instance['box_types']
    
    # Track available free space as a list of boxes (x, y, z, dx, dy, dz)
    # Start with the whole container as one free space
    free_spaces = [(0, 0, 0, container_dims[0], container_dims[1], container_dims[2])]
    
    placements = []
    
    # Sort box types by volume ascending (Small-First) to fill gaps
    # This is the opposite of the portfolio's standard 'Largest-First' bias.
    sorted_types = sorted(
        box_types.keys(),
        key=lambda bt: (box_types[bt]['dims'][0] * box_types[bt]['dims'][1] * box_types[bt]['dims'][2])
    )
    
    # Process until out of time or space
    while (time.time() - start_time) < (time_limit_s * 0.8):
        if not free_spaces:
            break
            
        # Pick the smallest free space (Best-Fit Space)
        free_spaces.sort(key=lambda s: s[3] * s[4] * s[5])
        space = free_spaces.pop(0)
        sx, sy, sz, sdx, sdy, sdz = space
        
        placed = False
        for bt in sorted_types:
            if tools['used_count'](placements, bt) >= box_types[bt]['count']:
                continue
                
            # Check all legal orientations for this box type
            for orient in tools['box_orientations'](bt):
                v, hswap, ox, oy, oz = orient
                
                # Check if box fits in current sub-space
                if ox <= sdx and oy <= sdy and oz <= sdz:
                    placement = {
                        'box_type': bt,
                        'container_id': 0,
                        'x': sx,
                        'y': sy,
                        'z': sz,
                        'v': v,
                        'hswap': hswap
                    }
                    placements.append(placement)
                    
                    # Split the remaining space into up to 3 new sub-spaces
                    # Splitting logic: cut along the longest remaining axis
                    if sdx - ox > 0:
                        free_spaces.append((sx + ox, sy, sz, sdx - ox, sdy, sdz))
                    if sdy - oy > 0:
                        free_spaces.append((sx, sy + oy, sz, ox, sdy - oy, sdz))
                    if sdz - oz > 0:
                        free_spaces.append((sx, sy, sz + oz, ox, oy, sdz - oz))
                    
                    placed = True
                    break
            if placed:
                break
        
        # If no box fits in this space, discard it (already popped)
        
    # Final cleanup: check if we have room for any more using standard corner-pack 
    # to utilize the leftover volume that the space-splitter might have missed.
    if (time.time() - start_time) < (time_limit_s * 0.95):
        remaining_placements = tools['corner_pack_3d'](allow_rotation=True)
        # Merge, but only keep if valid (no overlaps)
        for p in remaining_placements:
            if not tools['overlap_3d'](placements, p):
                placements.append(p)
                
    return tools['make_solution'](placements)