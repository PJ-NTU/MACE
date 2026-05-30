# MACE evolved heuristic 09/10 for problem: unconstrained_guillotine_cutting
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Simulated Annealing approach using a 'Sequence-Pair' inspired representation
    for relative positioning of rectangles. Most portfolio members use greedy
    MaxRects or shelf-packing; this approach treats the relative order of
    pieces as the primary decision variable and uses a constructive decoder
    that is not strictly shelf-based, allowing for more complex, non-guillotine
    interlocking patterns that local search can optimize.
    """
    start_time = time.time()
    pieces = instance['pieces']
    stock_w, stock_h = tools['stock_dims']()
    allow_rot = instance.get('allow_rotation', False)
    
    # Decouple piece selection from placement order
    # Current sequence represents the order of insertion
    ids = list(pieces.keys())
    
    def decode(sequence):
        # A simple 'Bottom-Left' placement decoder that validates against 
        # the global overlap tool implicitly or explicitly.
        placements = []
        occupied = []
        
        for pid in sequence:
            p = pieces[pid]
            orientations = [0, 1] if allow_rot else [0]
            
            best_pos = None
            best_o = 0
            
            # Scan potential positions based on existing boundaries
            # This is fundamentally different from MaxRects' list of free space
            candidates = [(0, 0)]
            for (ox, oy, ol, ow) in occupied:
                candidates.append((ox + ol, oy))
                candidates.append((ox, oy + ow))
            
            for x, y in candidates:
                for o in orientations:
                    l, w = (p['l'], p['w']) if o == 0 else (p['w'], p['l'])
                    if x + l <= stock_w and y + w <= stock_h:
                        # Collision check
                        if not any(not (x + l <= ox or x >= ox + ol or y + w <= oy or y >= oy + ow) 
                                   for (ox, oy, ol, ow) in occupied):
                            best_pos = (x, y, l, w)
                            best_o = o
                            break
                if best_pos: break
            
            if best_pos:
                placements.append({'piece_id': pid, 'x': best_pos[0], 'y': best_pos[1], 'orientation': best_o})
                occupied.append(best_pos)
        return placements

    # Simulated Annealing
    current_seq = list(ids)
    random.shuffle(current_seq)
    current_sol = decode(current_seq)
    current_val = sum(pieces[p['piece_id']]['value'] for p in current_sol)
    
    best_sol = current_sol
    best_val = current_val
    
    temp = 1.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit_s * 0.95:
        # Move: swap two elements in the sequence
        i, j = random.sample(range(len(current_seq)), 2)
        new_seq = list(current_seq)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        
        # Decode and evaluate
        cand_sol = decode(new_seq)
        cand_val = sum(pieces[p['piece_id']]['value'] for p in cand_sol)
        
        # Metropolis acceptance criterion
        delta = cand_val - current_val
        if delta > 0 or (temp > 0 and random.random() < np.exp(delta / (temp + 1e-9))):
            current_seq = new_seq
            current_val = cand_val
            current_sol = cand_sol
            
            if cand_val > best_val:
                best_val = cand_val
                best_sol = cand_sol
        
        temp *= cooling_rate
        
    return {"placements": best_sol}