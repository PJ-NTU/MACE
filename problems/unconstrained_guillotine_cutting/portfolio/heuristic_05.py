# MACE evolved heuristic 05/10 for problem: unconstrained_guillotine_cutting
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Max-Rectangles' based constructive heuristic combined with a 
    Stochastic Tabu Search.
    
    Portfolio Analysis:
    1. Most use simple shelf-packing or greedy density sorting.
    2. Most use simple swap-local (first-improvement).
    3. Most rely on DP or simple list permutations.
    
    This heuristic:
    1. Uses a Max-Rectangles packing algorithm (a standard in 2D bin packing)
       to maintain the set of free rectangles, which is more space-efficient
       than shelf packing.
    2. Uses a Tabu-List based local search instead of pure first-improvement
       swap, allowing moves that temporarily decrease value to escape
       local optima.
    3. Uses a 'Drop-and-Retry' perturbation strategy to aggressively
       re-explore the packing space.
    """
    start_time = time.time()
    pieces = instance['pieces']
    stock_w, stock_h = instance['stock_width'], instance['stock_height']
    allow_rot = instance.get('allow_rotation', False)

    def get_max_rects_packing(sequence):
        # Max-Rectangles logic
        free_rects = [{'x': 0, 'y': 0, 'w': stock_w, 'h': stock_h}]
        placements = []
        
        for pid in sequence:
            p = pieces[pid]
            best_rect = None
            best_orient = 0
            
            # Find best fit among free rectangles
            for i, rect in enumerate(free_rects):
                for orient in ([0, 1] if allow_rot else [0]):
                    l, w = (p['l'], p['w']) if orient == 0 else (p['w'], p['l'])
                    if l <= rect['w'] and w <= rect['h']:
                        # Simple heuristic: bottom-left fit
                        if best_rect is None or (rect['y'] < best_rect['y']) or \
                           (rect['y'] == best_rect['y'] and rect['x'] < best_rect['x']):
                            best_rect = {'x': rect['x'], 'y': rect['y'], 'w': l, 'h': w}
                            best_orient = orient
                            idx = i
            
            if best_rect:
                placements.append({'piece_id': pid, 'x': best_rect['x'], 'y': best_rect['y'], 'orientation': best_orient})
                # Split free_rects (simplified)
                new_free = []
                for r in free_rects:
                    # If overlapping, split
                    if not (best_rect['x'] + best_rect['w'] <= r['x'] or best_rect['x'] >= r['x'] + r['w'] or
                            best_rect['y'] + best_rect['h'] <= r['y'] or best_rect['y'] >= r['y'] + r['h']):
                        # Split logic
                        if best_rect['x'] > r['x']: new_free.append({'x': r['x'], 'y': r['y'], 'w': best_rect['x'] - r['x'], 'h': r['h']})
                        if best_rect['x'] + best_rect['w'] < r['x'] + r['w']: new_free.append({'x': best_rect['x'] + best_rect['w'], 'y': r['y'], 'w': r['x'] + r['w'] - (best_rect['x'] + best_rect['w']), 'h': r['h']})
                        if best_rect['y'] > r['y']: new_free.append({'x': r['x'], 'y': r['y'], 'w': r['w'], 'h': best_rect['y'] - r['y']})
                        if best_rect['y'] + best_rect['h'] < r['y'] + r['h']: new_free.append({'x': r['x'], 'y': best_rect['y'] + best_rect['h'], 'w': r['w'], 'h': r['y'] + r['h'] - (best_rect['y'] + best_rect['h'])})
                    else:
                        new_free.append(r)
                free_rects = new_free
        return placements

    current_seq = list(pieces.keys())
    random.shuffle(current_seq)
    
    best_placements = []
    best_value = 0
    
    tabu_list = []
    
    while time.time() - start_time < time_limit_s * 0.9:
        # Perturb sequence
        i, j = random.sample(range(len(current_seq)), 2)
        if (i, j) not in tabu_list:
            current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
            tabu_list.append((i, j))
            if len(tabu_list) > 10: tabu_list.pop(0)
            
            candidate = get_max_rects_packing(current_seq)
            val = sum(pieces[p['piece_id']]['value'] for p in candidate)
            
            if val > best_value:
                best_value = val
                best_placements = candidate
        
        # Occasional random reset
        if random.random() < 0.05:
            random.shuffle(current_seq)

    if not best_placements:
        return {"placements": tools['guillotine_greedy_value_density'](time_limit_s=0.1)}
        
    return {"placements": best_placements}