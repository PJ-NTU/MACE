# MACE evolved heuristic 08/10 for problem: unconstrained_guillotine_cutting
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust hybrid packing heuristic:
    1. Uses the Gilmore-Gomory DP for optimal sub-structure identification.
    2. Uses a randomized greedy constructive heuristic with localized search.
    3. Employs a 'Best-Improvement' meta-heuristic to iterate over packing sequences.
    """
    start_time = time.time()
    pieces = instance['pieces']
    stock_w, stock_h = instance['stock_width'], instance['stock_height']
    allow_rot = instance.get('allow_rotation', False)
    
    # 1. Warm start with DP (if time allows) or Greedy
    best_sol = {"placements": []}
    best_val = 0
    
    # Attempt DP for high-quality baseline
    try:
        dp_res = tools['gilmore_gomory_dp'](time_limit_s=time_limit_s * 0.3)
        if dp_res['placements']:
            best_sol = {"placements": dp_res['placements']}
            best_val = sum(pieces[p['piece_id']]['value'] for p in best_sol['placements'])
    except:
        pass

    # 2. Refine with Randomized Greedy + Local Search
    def get_packing(sequence):
        free = [{'x': 0, 'y': 0, 'w': stock_w, 'h': stock_h}]
        placements = []
        for pid in sequence:
            p = pieces[pid]
            best_r = None
            best_o = 0
            for i, r in enumerate(free):
                for o in ([0, 1] if allow_rot else [0]):
                    pw, ph = (p['l'], p['w']) if o == 0 else (p['w'], p['l'])
                    if pw <= r['w'] and ph <= r['h']:
                        if best_r is None or (r['y'] < best_r['y']) or (r['y'] == best_r['y'] and r['x'] < best_r['x']):
                            best_r = {'x': r['x'], 'y': r['y'], 'w': pw, 'h': ph}
                            best_o = o
            if best_r:
                placements.append({'piece_id': pid, 'x': best_r['x'], 'y': best_r['y'], 'orientation': best_o})
                new_free = []
                for r in free:
                    if not (best_r['x'] >= r['x'] + r['w'] or best_r['x'] + best_r['w'] <= r['x'] or
                            best_r['y'] >= r['y'] + r['h'] or best_r['y'] + best_r['h'] <= r['y']):
                        if best_r['x'] > r['x']: new_free.append({'x': r['x'], 'y': r['y'], 'w': best_r['x'] - r['x'], 'h': r['h']})
                        if best_r['x'] + best_r['w'] < r['x'] + r['w']: new_free.append({'x': best_r['x'] + best_r['w'], 'y': r['y'], 'w': r['x'] + r['w'] - (best_r['x'] + best_r['w']), 'h': r['h']})
                        if best_r['y'] > r['y']: new_free.append({'x': r['x'], 'y': r['y'], 'w': r['w'], 'h': best_r['y'] - r['y']})
                        if best_r['y'] + best_r['h'] < r['y'] + r['h']: new_free.append({'x': r['x'], 'y': best_r['y'] + best_r['h'], 'w': r['w'], 'h': r['y'] + r['h'] - (best_r['y'] + best_r['h'])})
                    else:
                        new_free.append(r)
                free = new_free
        return placements

    # Iterative Search
    ids = list(pieces.keys())
    # Sort by value density as a strong prior
    ids.sort(key=lambda pid: pieces[pid]['value'] / (pieces[pid]['l'] * pieces[pid]['w']), reverse=True)
    
    while time.time() - start_time < time_limit_s * 0.95:
        # Perturb sequence
        seq = list(ids)
        # Apply random swaps to explore neighborhood
        for _ in range(random.randint(1, 5)):
            a, b = random.sample(range(len(seq)), 2)
            seq[a], seq[b] = seq[b], seq[a]
            
        cand = get_packing(seq)
        val = sum(pieces[p['piece_id']]['value'] for p in cand)
        if val > best_val:
            best_val = val
            best_sol = {"placements": cand}
            
    # Final Polish
    try:
        final_placements = tools['apply_swap_local'](best_sol['placements'], t_limit=max(0.01, (time_limit_s - (time.time() - start_time)) * 0.5))
        return {"placements": final_placements}
    except:
        return best_sol