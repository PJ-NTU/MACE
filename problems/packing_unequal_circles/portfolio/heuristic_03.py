# MACE evolved heuristic 03/10 for problem: packing_unequal_circles
import time
import math
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Simulated Annealing on the number of packed circles' heuristic.
    
    Differing from the portfolio:
    1. Instead of 'construct-then-compact' (which is stuck in a single prefix length
       or local geometry), this uses a cooling-based acceptance criterion for 
       the number of circles packed.
    2. It explicitly relaxes the prefix constraint by 'ejection chain' moves: 
       if a new circle cannot fit, it randomly removes a circle and attempts 
       to re-insert at a different position, potentially allowing a larger set.
    """
    start_time = time.time()
    n = instance["n"]
    R = instance["R"]
    
    # Start with a full-prefix construct
    placements = tools['front_packing_construct'](num_angles=30)
    best_placements = list(placements)
    best_count = sum(1 for p in placements if p is not None)
    
    current_placements = list(placements)
    temp = 1.0
    
    while time.time() - start_time < time_limit_s * 0.9:
        # 1. Ejection move: remove a random circle (if any) to create slack
        packed_indices = [i for i, p in enumerate(current_placements) if p is not None]
        if not packed_indices:
            current_placements = tools['front_packing_construct'](num_angles=20)
            continue
            
        # Select a circle to eject (prefer larger/later ones to make space)
        eject_idx = random.choice(packed_indices)
        current_placements[eject_idx] = None
        
        # 2. Attempt to fill the gap or extend
        # Try to re-place the ejected circle or the next one in sequence
        # We use a randomized search in the remaining space
        for _ in range(3):
            extended = tools['try_place_next'](current_placements, num_angles=40)
            if extended:
                current_placements = extended
        
        # 3. Acceptance criterion
        count = sum(1 for p in current_placements if p is not None)
        delta = count - best_count
        
        if delta > 0 or (temp > 0 and random.random() < math.exp(delta / temp)):
            if count > best_count:
                best_count = count
                best_placements = list(current_placements)
        else:
            # Revert to best if move rejected
            current_placements = list(best_placements)
            
        # 4. Cooling
        temp *= 0.995
        
        # Periodic compacting to keep the 'best' solution tight
        if random.random() < 0.1:
            best_placements = tools['apply_local_shift'](best_placements, t_limit_s=0.1)
            current_placements = list(best_placements)
            
    return {"coords": tools['to_coords'](best_placements)}