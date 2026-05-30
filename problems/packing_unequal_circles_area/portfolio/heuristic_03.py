# MACE evolved heuristic 03/10 for problem: packing_unequal_circles_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust Greedy + Local Search strategy for packing unequal circles.
    
    Weakness of previous SA: 
    1. Purely random toggles often lead to infeasible states or tiny, ineffective shifts.
    2. The search space is dominated by the packing order; SA spends too much time 
       re-packing small circles when the real gain is in identifying the optimal 
       subset of large circles.
       
    Redesign:
    1. Use 'greedy_by_area_first' as a strong, high-baseline construction.
    2. Implement a 'Swap' focused LNS: prioritize swapping out small circles 
       for larger ones that are currently unpacked.
    3. Use 'try_relocate_circle' to compact the current set, creating 'holes' 
       to allow larger circles to fit.
    """
    start_time = time.time()
    n = instance['n']
    radii = instance['radii']
    
    # Sort indices by radius descending to facilitate high-value greedy starts
    sorted_indices = sorted(range(n), key=lambda i: radii[i], reverse=True)
    
    # Construction: Start with a heavy, high-value packing
    coords = tools['greedy_by_area_first'](attempts_per_circle=800)
    best_coords = list(coords)
    best_area = tools['total_area'](coords)
    
    # LNS Loop
    # Focus on:
    # 1. Swapping a smaller packed circle for a larger unpacked one.
    # 2. Relocating to create space.
    
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.90:
        iteration += 1
        
        # Strategy: Attempt to improve the current set by swapping
        packed = tools['placed_indices'](coords)
        unpacked = [i for i in range(n) if not tools['is_placed'](i, coords)]
        
        if not unpacked:
            break
            
        # Try to swap a small packed circle for a large unpacked one
        if packed and unpacked:
            out_i = random.choice(packed)
            in_i = random.choice(unpacked)
            
            # Only swap if the incoming circle is larger (area gain)
            if radii[in_i] > radii[out_i]:
                new_coords = tools['try_swap_in_out'](coords, out_i, in_i, attempts=600)
                if new_coords:
                    coords = new_coords
            else:
                # Otherwise, try to fit the large one anyway
                new_coords = tools['try_add_circle'](coords, in_i, attempts=600)
                if new_coords:
                    coords = new_coords
        
        # Periodic compaction
        if iteration % 5 == 0 and packed:
            idx = random.choice(packed)
            moved = tools['try_relocate_circle'](coords, idx, attempts=400)
            if moved:
                coords = moved
        
        # Track best
        current_area = tools['total_area'](coords)
        if current_area > best_area:
            best_area = current_area
            best_coords = list(coords)
            
        # If we reach a point where we can't improve for a while, restart from greedy
        if iteration % 200 == 0 and random.random() < 0.2:
            coords = tools['greedy_pack_in_order'](random.sample(range(n), n), attempts_per_circle=300)
            
    return {"coords": best_coords}