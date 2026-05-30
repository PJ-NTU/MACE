# MACE evolved heuristic 09/10 for problem: packing_unequal_circles_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An Adaptive Iterated Local Search (ILS) heuristic for unequal circle packing.
    
    Combines the strengths of population-based diversity (h_a) with the targeted 
    local improvement (h_b). It uses:
    1. A multi-start greedy construction (area-first) to ensure a strong baseline.
    2. A 'Perturb-Improve' cycle:
       - Perturbation: Randomly remove a cluster of circles (destroy).
       - Improvement: Hill-climbing via greedy insertion and swap moves.
    3. Adaptive parameter tuning: Increases search intensity as time allows.
    """
    start_time = time.time()
    n = tools['num_circles']()
    radii = instance['radii']
    
    # Baseline: Start with the best-known greedy approach
    best_coords = tools['greedy_by_area_first'](attempts_per_circle=400)
    best_area = tools['total_area'](best_coords)
    
    # Local improvement loop
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.92:
        iteration += 1
        
        # 1. Perturbation (Ruin)
        # Randomly remove 15-30% of packed circles to escape local optima
        current_coords = list(best_coords)
        packed = [i for i in range(n) if tools['is_placed'](i, current_coords)]
        
        if len(packed) > 2:
            num_to_remove = max(1, len(packed) // 4)
            to_remove = random.sample(packed, num_to_remove)
            for idx in to_remove:
                current_coords = tools['unplace_circle'](current_coords, idx)
        
        # 2. Re-construction (Recreate)
        # Attempt to fill the gaps with remaining circles, prioritizing area
        unpacked = [i for i in range(n) if not tools['is_placed'](i, current_coords)]
        random.shuffle(unpacked)
        unpacked.sort(key=lambda i: radii[i], reverse=True)
        
        for i in unpacked:
            temp = tools['try_add_circle'](current_coords, i, attempts=200)
            if temp:
                current_coords = temp
        
        # 3. Local Search (Improvement)
        # Try to swap small circles for larger unused ones
        unpacked = [i for i in range(n) if not tools['is_placed'](i, current_coords)]
        packed = [i for i in range(n) if tools['is_placed'](i, current_coords)]
        
        if unpacked and packed:
            out_i = random.choice(packed)
            in_i = max(unpacked, key=lambda i: radii[i])
            if radii[in_i] > radii[out_i]:
                temp = tools['try_swap_in_out'](current_coords, out_i, in_i, attempts=300)
                if temp:
                    current_coords = temp
        
        # 4. Acceptance
        current_area = tools['total_area'](current_coords)
        if current_area > best_area:
            best_area = current_area
            best_coords = current_coords
        
        # Periodic compaction: try to move a random circle to a better spot
        if iteration % 10 == 0:
            packed = [i for i in range(n) if tools['is_placed'](i, best_coords)]
            if packed:
                target = random.choice(packed)
                temp = tools['try_relocate_circle'](best_coords, target, attempts=150)
                if temp:
                    best_coords = temp
                    best_area = tools['total_area'](best_coords)
                    
    return {"coords": best_coords}