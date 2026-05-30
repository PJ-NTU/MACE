# MACE evolved heuristic 01/10 for problem: packing_unequal_circles
import time
import math
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the unequal circle packing problem using an iterative construction 
    and local search refinement process.
    """
    start_time = time.time()
    n = tools['n_circles']()
    
    # 1. Start with a good greedy baseline
    best_placements = tools['front_packing_construct'](num_angles=48)
    
    # Calculate initial number of packed circles
    def get_num_packed(placements):
        count = 0
        for p in placements:
            if p is not None:
                count += 1
            else:
                break
        return count

    best_count = get_num_packed(best_placements)
    
    # 2. Main optimization loop
    # Alternate between compacting the current layout and trying to add another circle
    while time.time() - start_time < time_limit_s * 0.9:
        # Try to compact the current layout to create space
        current_placements = tools['apply_local_shift'](
            best_placements, 
            t_limit_s=min(0.5, time_limit_s * 0.1),
            seed=random.randint(0, 10000)
        )
        
        # Try to add the next circle
        new_placements = tools['try_place_next'](
            current_placements, 
            num_angles=72, 
            grid_steps=40
        )
        
        if new_placements:
            count = get_num_packed(new_placements)
            if count > best_count:
                best_placements = new_placements
                best_count = count
            else:
                # If we didn't increase the count, maybe we found a better 
                # (more compact) version of the same count, keep it.
                best_placements = new_placements
        else:
            # If we couldn't add, try a random restart or continue shifting
            # to explore different configurations for the same number of circles
            if random.random() < 0.2:
                best_placements = tools['front_packing_construct'](num_angles=36)
                best_count = get_num_packed(best_placements)
                
    # 3. Final formatting
    coords = tools['to_coords'](best_placements)
    return {"coords": coords}