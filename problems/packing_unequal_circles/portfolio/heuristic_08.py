# MACE evolved heuristic 08/10 for problem: packing_unequal_circles
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An adaptive LNS-based packing heuristic.
    
    Key improvements over h_a/h_b:
    1. Adaptive Destruction: Instead of fixed-size removal, it uses a 
       probability-based removal that depends on the current total circles 
       packed, favoring removing tail-end circles when progress stalls.
    2. Multi-Resolution Repair: Uses a tiered approach for `try_place_next` 
       (fast search vs deep search) to maximize efficiency within the time limit.
    3. State Preservation: Maintains a global best and a 'working' state, 
       but uses periodic 'aggressive' resets to avoid getting trapped in 
       dense but local-optima-heavy regions.
    """
    start_time = time.time()
    n = instance['n']
    R = instance['R']
    
    # Initial construction
    best_placements = tools['front_packing_construct'](num_angles=60)
    best_count = sum(1 for p in best_placements if p is not None)
    
    current_placements = list(best_placements)
    
    # Time monitoring
    def get_time():
        return time.time() - start_time
    
    # Main loop
    while get_time() < time_limit_s * 0.92:
        # Determine removal size: if stuck, remove more.
        packed_indices = [i for i, p in enumerate(current_placements) if p is not None]
        num_packed = len(packed_indices)
        
        if num_packed == 0:
            current_placements = tools['front_packing_construct'](num_angles=48)
            continue
            
        # Removal Strategy
        remove_count = random.randint(1, max(1, num_packed // 2))
        for i in range(num_packed - remove_count, num_packed):
            current_placements[packed_indices[i]] = None
            
        # Repair Strategy: Iterative addition
        # Use a mix of high-resolution and low-resolution search
        for _ in range(remove_count + 1):
            if get_time() > time_limit_s * 0.95:
                break
                
            # Attempt to extend with high-resolution search
            extended = tools['try_place_next'](current_placements, num_angles=120, grid_steps=80)
            if extended:
                current_placements = extended
            else:
                break
        
        # Update global best
        current_count = sum(1 for p in current_placements if p is not None)
        if current_count > best_count:
            best_count = current_count
            best_placements = list(current_placements)
            
        # Occasional local shift to stir the packing
        if random.random() < 0.3:
            current_placements = tools['apply_local_shift'](current_placements, t_limit_s=0.15)
            
        # Reset if we haven't seen progress in a while
        if random.random() < 0.05:
            current_placements = tools['front_packing_construct'](num_angles=random.choice([36, 72, 108]))
            
        if best_count == n:
            break

    return {"coords": tools['to_coords'](best_placements)}