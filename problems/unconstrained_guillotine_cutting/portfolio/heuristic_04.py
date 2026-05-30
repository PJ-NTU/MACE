# MACE evolved heuristic 04/10 for problem: unconstrained_guillotine_cutting
import time
import random
import itertools

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A constructive heuristic based on 'Maximal Rectangles' (MaxRects) bin packing
    combined with a randomized best-fit descending strategy.
    
    The portfolio mostly relies on 'shelf' packing or DP-based guillotines. 
    MaxRects is a more flexible 2D packing strategy that maintains a set of 
    maximal free rectangles, which is generally more space-efficient than 
    simple shelf-based partitioning.
    """
    start_time = time.time()
    stock_w, stock_h = tools['stock_dims']()
    pieces = instance['pieces']
    allow_rot = instance.get('allow_rotation', False)
    
    def get_max_rects_pack(seed):
        rng = random.Random(seed)
        # Sort pieces by value density and introduce jitter for diversity
        piece_ids = list(pieces.keys())
        rng.shuffle(piece_ids)
        piece_ids.sort(key=lambda pid: pieces[pid]['value'] / (pieces[pid]['l'] * pieces[pid]['w']), reverse=True)
        
        # MaxRects maintains a list of all maximal empty rectangles
        free_rects = [{'x': 0, 'y': 0, 'w': stock_w, 'h': stock_h}]
        placements = []
        
        for pid in piece_ids:
            p = pieces[pid]
            opts = [(p['l'], p['w'], 0), (p['w'], p['l'], 1)] if allow_rot else [(p['l'], p['w'], 0)]
            rng.shuffle(opts)
            
            best_r = None
            best_orient = None
            
            # Find the free rectangle that best fits this piece (Best Area Fit)
            for r in free_rects:
                for pw, ph, orient in opts:
                    if pw <= r['w'] and ph <= r['h']:
                        if best_r is None or (r['w'] * r['h']) < (best_r['w'] * best_r['h']):
                            best_r = {'x': r['x'], 'y': r['y'], 'w': pw, 'h': ph}
                            best_orient = orient
            
            if best_r:
                placements.append({
                    'piece_id': pid,
                    'x': best_r['x'],
                    'y': best_r['y'],
                    'orientation': best_orient
                })
                
                # Split free_rects based on the new placement
                new_free = []
                for r in free_rects:
                    # If r overlaps with the new placement, split it
                    if not (best_r['x'] >= r['x'] + r['w'] or best_r['x'] + best_r['w'] <= r['x'] or
                            best_r['y'] >= r['y'] + r['h'] or best_r['y'] + best_r['h'] <= r['y']):
                        # Split into up to 4 smaller rectangles
                        if best_r['x'] > r['x']:
                            new_free.append({'x': r['x'], 'y': r['y'], 'w': best_r['x'] - r['x'], 'h': r['h']})
                        if best_r['x'] + best_r['w'] < r['x'] + r['w']:
                            new_free.append({'x': best_r['x'] + best_r['w'], 'y': r['y'], 'w': (r['x'] + r['w']) - (best_r['x'] + best_r['w']), 'h': r['h']})
                        if best_r['y'] > r['y']:
                            new_free.append({'x': r['x'], 'y': r['y'], 'w': r['w'], 'h': best_r['y'] - r['y']})
                        if best_r['y'] + best_r['h'] < r['y'] + r['h']:
                            new_free.append({'x': r['x'], 'y': best_r['y'] + best_r['h'], 'w': r['w'], 'h': (r['y'] + r['h']) - (best_r['y'] + best_r['h'])})
                    else:
                        new_free.append(r)
                free_rects = new_free
        return placements

    best_placements = []
    best_val = -1
    
    # Iterate with different seeds
    seed = 0
    while time.time() - start_time < time_limit_s * 0.9:
        candidate = get_max_rects_pack(seed)
        val = sum(pieces[p['piece_id']]['value'] for p in candidate)
        if val > best_val:
            best_val = val
            best_placements = candidate
        seed += 1
        
    return {"placements": best_placements}