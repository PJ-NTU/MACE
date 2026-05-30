# MACE evolved heuristic 10/10 for problem: packing_unequal_circles
import time
import math
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the unequal circle packing problem using an iterative construction 
    and local search refinement process with an adaptive construction strategy.
    """
    start_time = time.time()
    n = tools['n_circles']()
    
    # 1. Start with a more robust construction strategy: 
    # Use front_packing_construct as it is generally density-superior.
    best_placements = tools['front_packing_construct'](num_angles=60)
    
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
    # Modified: Increased exploration phase by allowing more aggressive 
    # re-constructions (restarts) when stuck, to escape local optima.
    while time.time() - start_time < time_limit_s * 0.9:
        # Try to compact the current layout to create space
        current_placements = tools['apply_local_shift'](
            best_placements, 
            t_limit_s=min(0.3, time_limit_s * 0.05),
            seed=random.randint(0, 10000)
        )
        
        # Try to add the next circle
        new_placements = tools['try_place_next'](
            current_placements, 
            num_angles=90, 
            grid_steps=50
        )
        
        if new_placements:
            count = get_num_packed(new_placements)
            if count > best_count:
                best_placements = new_placements
                best_count = count
            else:
                best_placements = new_placements
        else:
            # Increased restart probability to prevent stagnation on difficult instances
            if random.random() < 0.35:
                # Use a varying angle parameter to explore different packing patterns
                rand_angles = random.choice([24, 48, 72])
                best_placements = tools['front_packing_construct'](num_angles=rand_angles)
                best_count = get_num_packed(best_placements)
                
    # 3. Final formatting
    coords = tools['to_coords'](best_placements)
    return {"coords": coords}