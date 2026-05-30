# MACE evolved heuristic 04/10 for problem: packing_unequal_circles
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined heuristic for unequal circle packing.
    
    Diagnosis: The parent heuristic suffered from 'myopic compaction'—nudging 
    randomly often fails to create the specific geometric space required for the 
    next circle. It also lacked a systematic way to clear space near the boundary.
    
    Redesign:
    1. Multi-start with varying degrees of 'compactness' (grid vs front).
    2. 'Eviction & Try' strategy: When stuck, remove the last N circles to 
       aggressively compact the remaining M-N circles, then try to re-insert.
    3. Time-aware budget management: Focus on the most promising prefix lengths.
    """
    start_time = time.time()
    n = instance["n"]
    R = instance["R"]
    
    best_placements = [None] * n
    best_count = 0

    def get_count(placements):
        count = 0
        for p in placements:
            if p is not None:
                count += 1
            else:
                break
        return count

    def run_expansion(current, time_limit_local):
        nonlocal best_count, best_placements
        while time.time() - start_time < time_limit_local:
            next_p = tools['try_place_next'](current, num_angles=90, grid_steps=50)
            if next_p:
                current = next_p
                count = get_count(current)
                if count > best_count:
                    best_count = count
                    best_placements = list(current)
                if count == n:
                    return current
            else:
                break
        return current

    # Main loop
    iter_idx = 0
    while time.time() - start_time < time_limit_s * 0.92:
        # Strategy selection
        if iter_idx % 4 == 0:
            current = tools['front_packing_construct'](num_angles=60)
        elif iter_idx % 4 == 1:
            current = tools['prefix_grid_construct'](grid_steps=40)
        else:
            # Eviction: Start from a truncated version of the current best
            current = list(best_placements)
            cnt = get_count(current)
            if cnt > 2:
                # Remove 20% to 50% of the currently packed circles
                remove_n = max(1, int(cnt * 0.3))
                for i in range(cnt - remove_n, cnt):
                    current[i] = None
            else:
                current = [None] * n
        
        # Expand
        current = run_expansion(current, time_limit_s * 0.95)
        
        # Compaction: Only run if we have a decent number of circles
        cnt = get_count(current)
        if cnt > 1 and time.time() - start_time < time_limit_s * 0.9:
            # Aggressive local shift to tighten current layout
            current = tools['apply_local_shift'](
                current, 
                t_limit_s=min(0.4, (time_limit_s * 0.95 - (time.time() - start_time))), 
                delta=R * 0.05
            )
            # Try to extend again after compaction
            current = run_expansion(current, time_limit_s * 0.95)
            
        iter_idx += 1
        
        if best_count == n:
            break

    return {"coords": tools['to_coords'](best_placements)}