# MACE evolved heuristic 07/10 for problem: packing_unequal_circles
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved heuristic for Unequal Circle Packing.
    
    Diagnosis of parent:
    1. The 'shake' mechanism is too destructive and random, often destroying 
       well-packed centers without a clear path back to optimality.
    2. The reliance on 'apply_local_shift' is limited because it is only used 
       as a polishing step, not integrated into the core growth loop.
    3. The VNS logic lacks a strong local search phase that attempts to 
       compact the current set to force a 'try_place_next' success.
    
    Redesign:
    - Implements an Iterated Local Search (ILS) with a focus on 'Compaction-Growth cycles'.
    - Prioritizes 'front_packing_construct' as the core engine, as it outperforms 
      grid methods for irregular circles.
    - Uses a robust 'Compaction-Growth' loop:
        1. Start with a greedy construction.
        2. Attempt to add a new circle.
        3. If it fails, use 'apply_local_shift' to rearrange the existing packing
           to create space.
        4. If it succeeds, repeat.
    - Periodically restarts with randomized construction parameters to escape local optima.
    """
    start_time = time.time()
    n = instance["n"]
    
    best_placements = [None] * n
    best_count = 0
    
    def get_count(placements):
        return sum(1 for p in placements if p is not None)

    # Time budget management
    # Reserve 10% of time for potential final cleanup
    while time.time() - start_time < time_limit_s * 0.9:
        # 1. Randomized Construction Phase
        # Diversify starting points using different angle counts
        num_angles = random.choice([36, 72, 144])
        curr = tools['front_packing_construct'](num_angles=num_angles)
        
        # 2. Growth and Compaction Phase
        # Loop to try and grow the packing incrementally
        while True:
            # Try to add a circle
            extended = tools['try_place_next'](curr, num_angles=120, grid_steps=40)
            if extended:
                curr = extended
            else:
                # If we couldn't add, try to compact the current layout 
                # to create space for the next circle
                before_compaction = get_count(curr)
                if before_compaction == 0:
                    break
                
                # Use local shift to optimize existing packing
                curr = tools['apply_local_shift'](curr, t_limit_s=0.15)
                
                # Try to add again after compaction
                extended = tools['try_place_next'](curr, num_angles=180, grid_steps=60)
                if extended:
                    curr = extended
                else:
                    # No progress possible from this configuration
                    break
        
        # 3. Acceptance
        count = get_count(curr)
        if count > best_count:
            best_count = count
            best_placements = list(curr)
        
        # If we have packed everything, exit early
        if best_count == n:
            break
            
    return {"coords": tools['to_coords'](best_placements)}