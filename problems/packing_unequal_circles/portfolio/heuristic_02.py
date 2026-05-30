# MACE evolved heuristic 02/10 for problem: packing_unequal_circles
import time
import math
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the unequal circle packing problem using a combination of
    front-packing construction and iterative local-search compaction.
    """
    start_time = time.time()
    n = instance["n"]
    
    # 1. Start with a strong greedy construction
    placements = tools['front_packing_construct'](num_angles=48)
    
    # Identify how many are currently packed
    def get_packed_count(placements):
        count = 0
        for p in placements:
            if p is not None:
                count += 1
            else:
                break
        return count

    best_placements = list(placements)
    best_count = get_packed_count(best_placements)
    
    # 2. Iterative Improvement Loop
    # We attempt to compact the current layout to make room for more circles.
    while time.time() - start_time < time_limit_s * 0.9:
        # Try to add one more circle if possible
        new_placements = tools['try_place_next'](best_placements, num_angles=60, grid_steps=40)
        
        if new_placements:
            best_placements = new_placements
            best_count = get_packed_count(best_placements)
            if best_count == n:
                break
        else:
            # If we cannot add another, try to compact the current set
            # to create more "free space" at the margins or between circles.
            # We use the provided local shift tool.
            time_remaining = (time_limit_s * 0.9) - (time.time() - start_time)
            if time_remaining <= 0.1:
                break
            
            # Use local shift to optimize current layout
            best_placements = tools['apply_local_shift'](
                best_placements, 
                t_limit_s=min(0.5, time_remaining),
                delta=0.05 * instance['R']
            )
            
            # Double check feasibility after shift
            # (apply_local_shift is designed to return feasible configurations)
            pass

    return {"coords": tools['to_coords'](best_placements)}