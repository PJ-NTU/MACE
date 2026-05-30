# MACE evolved heuristic 01/10 for problem: unconstrained_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Implements a robust heuristic for the Unconstrained Guillotine Cutting problem.
    Uses a combination of greedy value-density construction and stochastic 
    local search (swap and shift) to maximize total value.
    """
    start_time = time.time()
    
    # Initial greedy construction
    best_placements = tools['guillotine_greedy_value_density'](time_limit_s=max(0.1, time_limit_s * 0.2))
    best_value = sum(instance['pieces'][p['piece_id']]['value'] for p in best_placements)
    
    # Iterative improvement phase
    # We attempt to improve the solution by replacing pieces with better ones 
    # and trying to fit new pieces into the gaps.
    
    def get_current_value(placements):
        return sum(instance['pieces'][p['piece_id']]['value'] for p in placements)

    # 1. First-improvement local search using the provided swap tool
    improved_placements = tools['apply_swap_local'](best_placements, t_limit=max(0.1, time_limit_s * 0.3))
    
    current_val = get_current_value(improved_placements)
    if current_val > best_value:
        best_value = current_val
        best_placements = improved_placements

    # 2. Randomized Hill Climbing (Attempt to add remaining pieces)
    # Identify unused pieces
    used_ids = {p['piece_id'] for p in best_placements}
    unused_ids = [pid for pid in instance['pieces'] if pid not in used_ids]
    random.shuffle(unused_ids)
    
    # Try to fit unused pieces into empty spaces
    # Since we don't have a sophisticated packing engine, we rely on the 
    # provided tools and a simple greedy attempt.
    
    while time.time() - start_time < time_limit_s * 0.9:
        # Pick a random unused piece
        if not unused_ids:
            break
            
        pid = unused_ids.pop()
        piece = instance['pieces'][pid]
        
        # Try to find a spot (simple scanning)
        # Note: This is an unconstrained problem, so we try to find a valid coordinate
        # by checking feasibility with the current set.
        placed = False
        sw, sh = tools['stock_dims']()
        
        # Try a few random coordinates to see if it fits
        for _ in range(20):
            x = random.randint(0, sw - piece['l'])
            y = random.randint(0, sh - piece['w'])
            
            test_placements = best_placements + [{
                'piece_id': pid,
                'x': x,
                'y': y,
                'orientation': 0
            }]
            
            is_f, _ = tools['is_feasible']({'placements': test_placements})
            if is_f:
                best_placements = test_placements
                best_value += piece['value']
                placed = True
                break
        
        if not placed and instance['allow_rotation']:
            # Try rotated
            for _ in range(20):
                x = random.randint(0, sw - piece['w'])
                y = random.randint(0, sh - piece['l'])
                test_placements = best_placements + [{
                    'piece_id': pid,
                    'x': x,
                    'y': y,
                    'orientation': 1
                }]
                is_f, _ = tools['is_feasible']({'placements': test_placements})
                if is_f:
                    best_placements = test_placements
                    best_value += piece['value']
                    break

    return {"placements": best_placements}