# MACE evolved heuristic 06/10 for problem: packing_unequal_circles
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved circle packing heuristic using a multi-start, state-aware approach.
    
    Diagnosis of parent:
    1. The parent relies on a single `front_packing_construct` which can get stuck in sub-optimal local basins.
    2. The `apply_local_shift` is non-deterministic and can destroy a good packing if not careful.
    3. The loop lacks a clear strategy for re-initiating after failure.
    
    Redesign:
    1. Use a multi-start strategy with varying construction parameters to find diverse initial packings.
    2. Prioritize "tightening" the packing by repeatedly shifting toward the center.
    3. Use a time-budgeted, aggressive local search that favors larger packing counts.
    """
    start_time = time.time()
    n = tools['n_circles']()
    
    def get_num_packed(placements):
        count = 0
        for p in placements:
            if p is not None:
                count += 1
            else:
                break
        return count

    best_placements = [None] * n
    best_count = 0

    # Strategy: Try different construction styles, then refine the best one found.
    # We rotate through different initialization methods to diversify.
    constructors = [
        lambda: tools['front_packing_construct'](num_angles=24),
        lambda: tools['front_packing_construct'](num_angles=48),
        lambda: tools['front_packing_construct'](num_angles=72),
        lambda: tools['prefix_grid_construct'](grid_steps=30),
        lambda: tools['prefix_grid_construct'](grid_steps=50)
    ]
    
    idx = 0
    while time.time() - start_time < time_limit_s * 0.8:
        # 1. Initialization / Restart
        current_placements = constructors[idx % len(constructors)]()
        idx += 1
        
        # 2. Attempt to add more to this specific construction
        improved = True
        while improved and time.time() - start_time < time_limit_s * 0.9:
            next_p = tools['try_place_next'](current_placements, num_angles=60, grid_steps=40)
            if next_p:
                current_placements = next_p
            else:
                # Try to wiggle to create space
                current_placements = tools['apply_local_shift'](
                    current_placements, 
                    t_limit_s=0.2, 
                    delta=0.05
                )
                # Re-check if we can add one now
                next_p = tools['try_place_next'](current_placements, num_angles=60, grid_steps=40)
                if next_p:
                    current_placements = next_p
                else:
                    improved = False
        
        # 3. Update global best
        current_count = get_num_packed(current_placements)
        if current_count > best_count:
            best_count = current_count
            best_placements = current_placements
            
        # If we reached the theoretical maximum, break early
        if best_count == n:
            break

    return {"coords": tools['to_coords'](best_placements)}