# MACE evolved heuristic 10/10 for problem: unconstrained_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved heuristic based on MaxRects with a more robust piece ordering strategy:
    incorporating a 'Value Density' heuristic that prioritizes pieces with high
    value-to-area ratios while also considering absolute value for larger pieces.
    """
    start_time = time.time()
    stock_w, stock_h = tools['stock_dims']()
    pieces = instance['pieces']
    allow_rot = instance.get('allow_rotation', False)
    
    def get_max_rects_pack(seed):
        rng = random.Random(seed)
        piece_ids = list(pieces.keys())
        # Sort pieces by a weighted score: value density + absolute value normalized
        # to ensure high-value pieces are prioritized even if slightly inefficient
        def score(pid):
            p = pieces[pid]
            area = p['l'] * p['w']
            return (p['value'] / area) * 1000 + p['value']
        
        piece_ids.sort(key=score, reverse=True)
        # Apply slight perturbation to the order for exploration
        if len(piece_ids) > 1:
            for i in range(min(len(piece_ids), 3)):
                swap_idx = rng.randint(i, min(len(piece_ids) - 1, i + 5))
                piece_ids[i], piece_ids[swap_idx] = piece_ids[swap_idx], piece_ids[i]
        
        free_rects = [{'x': 0, 'y': 0, 'w': stock_w, 'h': stock_h}]
        placements = []
        
        for pid in piece_ids:
            p = pieces[pid]
            opts = [(p['l'], p['w'], 0), (p['w'], p['l'], 1)] if allow_rot else [(p['l'], p['w'], 0)]
            
            best_r_idx = -1
            best_r = None
            best_orient = None
            min_area = float('inf')
            
            for i, r in enumerate(free_rects):
                for pw, ph, orient in opts:
                    if pw <= r['w'] and ph <= r['h']:
                        # Use BSSF (Best Short Side Fit) rule to keep remaining free space contiguous
                        area = r['w'] * r['h']
                        if area < min_area:
                            min_area = area
                            best_r = {'x': r['x'], 'y': r['y'], 'w': pw, 'h': ph}
                            best_orient = orient
                            best_r_idx = i
            
            if best_r:
                placements.append({
                    'piece_id': pid,
                    'x': best_r['x'],
                    'y': best_r['y'],
                    'orientation': best_orient
                })
                
                # Update free_rects: remove the used one and add sub-rectangles
                old_r = free_rects.pop(best_r_idx)
                new_free = []
                # Split remainder into two maximal rectangles (Guillotine-style split)
                # Split along the side that minimizes fragmentation
                if (old_r['w'] - best_r['w']) > (old_r['h'] - best_r['h']):
                    new_free.append({'x': old_r['x'] + best_r['w'], 'y': old_r['y'], 'w': old_r['w'] - best_r['w'], 'h': old_r['h']})
                    new_free.append({'x': old_r['x'], 'y': old_r['y'] + best_r['h'], 'w': best_r['w'], 'h': old_r['h'] - best_r['h']})
                else:
                    new_free.append({'x': old_r['x'], 'y': old_r['y'] + best_r['h'], 'w': old_r['w'], 'h': old_r['h'] - best_r['h']})
                    new_free.append({'x': old_r['x'] + best_r['w'], 'y': old_r['y'], 'w': old_r['w'] - best_r['w'], 'h': best_r['h']})
                
                # Filter out empty or invalid rectangles
                for nr in new_free:
                    if nr['w'] > 0 and nr['h'] > 0:
                        free_rects.append(nr)
        return placements

    best_placements = []
    best_val = -1
    seed = 0
    while time.time() - start_time < time_limit_s * 0.9:
        candidate = get_max_rects_pack(seed)
        val = sum(pieces[p['piece_id']]['value'] for p in candidate)
        if val > best_val:
            best_val = val
            best_placements = candidate
        seed += 1
        
    return {"placements": best_placements}