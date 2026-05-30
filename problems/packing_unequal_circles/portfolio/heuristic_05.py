# MACE evolved heuristic 05/10 for problem: packing_unequal_circles
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved heuristic: Focuses on iterative refinement through aggressive 
    local search compaction. Instead of randomized ejection, it uses a 
    'compact-then-extend' strategy. It prioritizes building the longest 
    feasible prefix using high-density front-packing and refines the 
    coordinates via local perturbations to maximize room for subsequent circles.
    """
    start_time = time.time()
    n = instance["n"]
    
    # 1. Initial Construction
    # Use front_packing as the base; it is superior to grid_construct for circle packing.
    placements = tools['front_packing_construct'](num_angles=60)
    
    best_placements = list(placements)
    best_count = sum(1 for p in placements if p is not None)
    
    # 2. Main Refinement Loop
    # We alternate between compacting the current layout and trying to extend the prefix.
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.95:
        iteration += 1
        
        # Periodically reset if stuck
        if iteration % 10 == 0:
            placements = tools['front_packing_construct'](num_angles=30 + (iteration % 30))
        
        # A. Compact: Move circles to create space for the next one
        # We give more time to the compactor as the density increases
        compaction_time = 0.15 if best_count < n * 0.5 else 0.4
        placements = tools['apply_local_shift'](placements, t_limit_s=compaction_time)
        
        # B. Extend: Try to add the next circle in the sequence
        # We try multiple times per compaction
        for _ in range(2):
            extended = tools['try_place_next'](placements, num_angles=90, grid_steps=40)
            if extended:
                placements = extended
                current_count = sum(1 for p in placements if p is not None)
                if current_count > best_count:
                    best_count = current_count
                    best_placements = list(placements)
            else:
                break
        
        # C. Local Search/Restart
        # If we reached max, stop early
        if best_count == n:
            break
            
        # Occasionally perform a slight perturbation to escape local optima
        if random.random() < 0.2:
            # Drop the last few placed circles to allow the compactor to re-arrange
            packed_indices = [i for i, p in enumerate(placements) if p is not None]
            if len(packed_indices) > 2:
                for idx in packed_indices[-2:]:
                    placements[idx] = None
    
    return {"coords": tools['to_coords'](best_placements)}