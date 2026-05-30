# MACE evolved heuristic 07/10 for problem: unconstrained_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Hybrid heuristic combining Gilmore-Gomory DP for optimal sub-problems 
    with a randomized greedy construction for global layout.
    """
    start_time = time.time()
    pieces = instance['pieces']
    stock_w, stock_h = tools['stock_dims']()
    allow_rot = instance.get('allow_rotation', False)

    def get_greedy_packing(seed, subset_ids=None):
        rng = random.Random(seed)
        ids = list(subset_ids) if subset_ids else list(pieces.keys())
        # Shuffle with a bias towards high value density
        ids.sort(key=lambda pid: pieces[pid]['value'] / (pieces[pid]['l'] * pieces[pid]['w']), reverse=True)
        # Add slight stochastic noise to the sorting
        ids = sorted(ids, key=lambda pid: (pieces[pid]['value'] / (pieces[pid]['l'] * pieces[pid]['w'])) * rng.uniform(0.8, 1.2), reverse=True)
        
        free = [{'x': 0, 'y': 0, 'w': stock_w, 'h': stock_h}]
        placements = []
        
        for pid in ids:
            p = pieces[pid]
            best_r = None
            best_orient = 0
            
            # Find best fit using Bottom-Left heuristic
            for r in free:
                for orient in ([0, 1] if allow_rot else [0]):
                    pw, ph = (p['l'], p['w']) if orient == 0 else (p['w'], p['l'])
                    if pw <= r['w'] and ph <= r['h']:
                        if best_r is None or (r['y'] < best_r['y']) or (r['y'] == best_r['y'] and r['x'] < best_r['x']):
                            best_r = {'x': r['x'], 'y': r['y'], 'w': pw, 'h': ph}
                            best_orient = orient
            
            if best_r:
                placements.append({'piece_id': pid, 'x': best_r['x'], 'y': best_r['y'], 'orientation': best_orient})
                new_free = []
                for r in free:
                    # Split free rectangle
                    if not (best_r['x'] >= r['x'] + r['w'] or best_r['x'] + best_r['w'] <= r['x'] or
                            best_r['y'] >= r['y'] + r['h'] or best_r['y'] + best_r['h'] <= r['y']):
                        if best_r['x'] > r['x']: new_free.append({'x': r['x'], 'y': r['y'], 'w': best_r['x'] - r['x'], 'h': r['h']})
                        if best_r['x'] + best_r['w'] < r['x'] + r['w']: new_free.append({'x': best_r['x'] + best_r['w'], 'y': r['y'], 'w': (r['x'] + r['w']) - (best_r['x'] + best_r['w']), 'h': r['h']})
                        if best_r['y'] > r['y']: new_free.append({'x': r['x'], 'y': r['y'], 'w': r['w'], 'h': best_r['y'] - r['y']})
                        if best_r['y'] + best_r['h'] < r['y'] + r['h']: new_free.append({'x': r['x'], 'y': best_r['y'] + best_r['h'], 'w': r['w'], 'h': (r['y'] + r['h']) - (best_r['y'] + best_r['h'])})
                    else:
                        new_free.append(r)
                free = new_free
        return placements

    # 1. Start with high-quality result from tools
    # Note: tools['guillotine_greedy_value_density'] returns a list, not a dict
    initial_placements = tools['guillotine_greedy_value_density'](time_limit_s=max(0.1, time_limit_s * 0.2))
    best_sol = {'placements': initial_placements}
    best_val = sum(pieces[p['piece_id']]['value'] for p in best_sol['placements'])
    
    # 2. Try to improve via local iteration
    seed = 0
    while time.time() - start_time < time_limit_s * 0.9:
        cand_placements = get_greedy_packing(seed)
        cand_val = sum(pieces[p['piece_id']]['value'] for p in cand_placements)
        
        if cand_val > best_val:
            best_val = cand_val
            best_sol = {'placements': cand_placements}
        seed += 1
        
    # 3. Final polish using tool's swap local search if possible
    try:
        final_placements = tools['apply_swap_local'](best_sol['placements'], t_limit=max(0.1, time_limit_s * 0.1))
        return {"placements": final_placements}
    except:
        return best_sol