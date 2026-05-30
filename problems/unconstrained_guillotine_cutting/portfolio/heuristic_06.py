# MACE evolved heuristic 06/10 for problem: unconstrained_guillotine_cutting
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher heuristic for the Unconstrained Guillotine Cutting problem.
    
    Hypothesis:
    - MaxRects (A-style) is superior for high-density instances or instances 
      with many small pieces where space partitioning needs to be aggressive 
      and non-guillotine-restricted.
    - LNS (B-style) is superior for sparse instances or those with large 
      valuable pieces where global optimization through destruction/repair 
      can escape local optima better than a one-pass construction.
    """
    start_time = time.time()
    pieces = instance['pieces']
    stock_w, stock_h = tools['stock_dims']()
    m = instance['m']
    
    # Calculate packing density metric
    total_area = sum(p['l'] * p['w'] for p in pieces.values())
    stock_area = stock_w * stock_h
    density = total_area / stock_area
    
    # Dispatch: High density -> MaxRects, Low density -> LNS
    # Also favor LNS if piece count is small (easier to shuffle/repair)
    if density > 0.6 or m > 50:
        return _solve_max_rects(instance, tools, time_limit_s)
    else:
        return _solve_lns(instance, tools, time_limit_s)

def _solve_max_rects(instance, tools, time_limit_s):
    start_time = time.time()
    stock_w, stock_h = tools['stock_dims']()
    pieces = instance['pieces']
    allow_rot = instance.get('allow_rotation', False)
    
    def get_max_rects_pack(seed):
        rng = random.Random(seed)
        piece_ids = list(pieces.keys())
        rng.shuffle(piece_ids)
        piece_ids.sort(key=lambda pid: pieces[pid]['value'] / (pieces[pid]['l'] * pieces[pid]['w']), reverse=True)
        
        free_rects = [{'x': 0, 'y': 0, 'w': stock_w, 'h': stock_h}]
        placements = []
        
        for pid in piece_ids:
            p = pieces[pid]
            opts = [(p['l'], p['w'], 0), (p['w'], p['l'], 1)] if allow_rot else [(p['l'], p['w'], 0)]
            # Priority to larger area pieces first in the search
            best_r = None
            best_orient = None
            
            for r in free_rects:
                for pw, ph, orient in opts:
                    if pw <= r['w'] and ph <= r['h']:
                        if best_r is None or (r['w'] * r['h']) < (best_r['w'] * best_r['h']):
                            best_r = {'x': r['x'], 'y': r['y'], 'w': pw, 'h': ph}
                            best_orient = orient
            
            if best_r:
                placements.append({'piece_id': pid, 'x': best_r['x'], 'y': best_r['y'], 'orientation': best_orient})
                new_free = []
                for r in free_rects:
                    if not (best_r['x'] >= r['x'] + r['w'] or best_r['x'] + best_r['w'] <= r['x'] or
                            best_r['y'] >= r['y'] + r['h'] or best_r['y'] + best_r['h'] <= r['y']):
                        if best_r['x'] > r['x']: new_free.append({'x': r['x'], 'y': r['y'], 'w': best_r['x'] - r['x'], 'h': r['h']})
                        if best_r['x'] + best_r['w'] < r['x'] + r['w']: new_free.append({'x': best_r['x'] + best_r['w'], 'y': r['y'], 'w': (r['x'] + r['w']) - (best_r['x'] + best_r['w']), 'h': r['h']})
                        if best_r['y'] > r['y']: new_free.append({'x': r['x'], 'y': r['y'], 'w': r['w'], 'h': best_r['y'] - r['y']})
                        if best_r['y'] + best_r['h'] < r['y'] + r['h']: new_free.append({'x': r['x'], 'y': best_r['y'] + best_r['h'], 'w': r['w'], 'h': (r['y'] + r['h']) - (best_r['y'] + best_r['h'])})
                    else: new_free.append(r)
                free_rects = new_free
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

def _solve_lns(instance, tools, time_limit_s):
    start_time = time.time()
    pieces = instance['pieces']
    stock_w, stock_h = tools['stock_dims']()
    allow_rot = instance.get('allow_rotation', False)
    
    def get_repair(current_placements):
        occupied = []
        for p in current_placements:
            l, w = tools['piece_dims'](p['piece_id'])
            if p['orientation'] == 1: l, w = w, l
            occupied.append((p['x'], p['y'], l, w))
        unused = [pid for pid in pieces if pid not in [p['piece_id'] for p in current_placements]]
        random.shuffle(unused)
        points = [(0, 0)] + [(ox + ol, oy) for (ox, oy, ol, ow) in occupied] + [(ox, oy + ow) for (ox, oy, ol, ow) in occupied]
        for pid in unused:
            p = pieces[pid]
            for px, py in points:
                for orient in ([1, 0] if allow_rot else [0]):
                    l, w = (p['w'], p['l']) if orient == 1 else (p['l'], p['w'])
                    if px + l <= stock_w and py + w <= stock_h:
                        if not any(not (px + l <= ox or px >= ox + ol or py + w <= oy or py >= oy + ow) for (ox, oy, ol, ow) in occupied):
                            current_placements.append({'piece_id': pid, 'x': px, 'y': py, 'orientation': orient})
                            occupied.append((px, py, l, w))
                            points.extend([(px + l, py), (px, py + w)])
                            break
        return current_placements

    best_placements = tools['guillotine_greedy_value_density'](time_limit_s=0.1)
    best_value = sum(pieces[p['piece_id']]['value'] for p in best_placements)
    while time.time() - start_time < time_limit_s * 0.95:
        destroy_count = random.randint(1, max(1, len(best_placements) // 2))
        candidate = random.sample(best_placements, len(best_placements) - destroy_count)
        candidate = get_repair(candidate)
        cand_val = sum(pieces[p['piece_id']]['value'] for p in candidate)
        if cand_val > best_value:
            best_value = cand_val
            best_placements = candidate
    return {"placements": best_placements}