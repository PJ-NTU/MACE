# MACE evolved heuristic 09/10 for problem: packing_unequal_circles
import time
import math
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust hybrid heuristic combining aggressive LNS destruction with 
    simulated annealing-inspired acceptance and intensive local compaction.
    """
    start_time = time.time()
    n = instance['n']
    R = instance['R']
    
    # 1. Initialization
    best_placements = tools['front_packing_construct'](num_angles=60)
    best_count = sum(1 for p in best_placements if p is not None)
    current_placements = list(best_placements)
    
    # Annealing parameters
    temp = 1.0
    cooling_rate = 0.99
    
    # Main Loop
    while time.time() - start_time < time_limit_s * 0.92:
        packed_indices = [i for i, p in enumerate(current_placements) if p is not None]
        
        if not packed_indices:
            current_placements = tools['front_packing_construct'](num_angles=36)
            continue
            
        # 2. Adaptive Destruction
        # Instead of just removing the tail (h_b) or a random one (h_a), 
        # we remove a block of circles to create a significant "hole" 
        # for potential reorganization.
        num_packed = len(packed_indices)
        remove_count = max(1, int(num_packed * 0.4))
        start_remove_idx = max(0, num_packed - remove_count)
        
        for i in range(start_remove_idx, num_packed):
            current_placements[packed_indices[i]] = None
            
        # 3. Intensive Repair and Compaction
        # Use high-resolution try_place_next to attempt to fill the hole
        # and potentially add new circles.
        for _ in range(remove_count + 1):
            extended = tools['try_place_next'](current_placements, num_angles=120, grid_steps=40)
            if extended:
                current_placements = extended
            else:
                break
        
        # 4. Local Shift Compaction
        # Apply compaction to tighten the packing, increasing the probability
        # that the next 'try_place_next' cycle succeeds.
        current_placements = tools['apply_local_shift'](current_placements, t_limit_s=0.15)
        
        # 5. Acceptance Criterion
        current_count = sum(1 for p in current_placements if p is not None)
        delta = current_count - best_count
        
        if delta > 0 or (temp > 0 and random.random() < math.exp(delta / (temp + 1e-9))):
            if current_count > best_count:
                best_count = current_count
                best_placements = list(current_placements)
        else:
            # Revert to last best known state
            current_placements = list(best_placements)
            
        # Periodic restart to avoid deep local optima
        if random.random() < 0.05:
            current_placements = tools['front_packing_construct'](num_angles=40)
            
        temp *= cooling_rate
        
    return {"coords": tools['to_coords'](best_placements)}