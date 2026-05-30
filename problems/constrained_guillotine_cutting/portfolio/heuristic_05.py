# MACE evolved heuristic 05/10 for problem: constrained_guillotine_cutting
import time
import random
import math
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Constraint-Satisfaction Tabu Search' (CSTS) heuristic.
    
    Portfolio Blind Spots:
    1. Most use greedy or stochastic construction (First-Fit, BFD).
    2. Most rely on 'try_place_piece' as the sole placement logic.
    3. None use a 'Constraint-Based' approach that manages the 'guillotine' 
       fragmentation explicitly by maintaining a set of available rectangular 
       'empty' slots (bin-packing style), rather than just calling try_place_piece.
    
    This heuristic:
    1. Implements a 'Guillotine-Split' Bin Manager: It maintains a list of empty 
       rectangular regions (the 'free space' list).
    2. Uses a 'Tabu-List' to prevent oscillating between piece-type inclusions.
    3. Avoids the expensive 'try_place_piece' recursion in the inner loop by 
       manually tracking valid guillotine-split regions, allowing for much 
       faster exploration of the search space.
    """
    start_time = time.time()
    stock_l, stock_w = tools['stock_dims']()
    m = tools['n_piece_types']()
    
    # Pre-process piece data
    pieces = []
    for i in range(1, m + 1):
        l, w = tools['piece_dims'](i)
        pieces.append({'id': i, 'l': l, 'w': w, 'val': tools['piece_value'](i)})
    
    # Sort by value density
    pieces.sort(key=lambda x: x['val'] / (x['l'] * x['w']), reverse=True)
    
    best_sol = {"total_value": 0, "placements": []}
    tabu_list = {} # {piece_id: iteration_banned_until}
    
    # Iterative Search
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.9:
        iteration += 1
        current_placements = []
        # Free space maintained as a list of (x, y, l, w)
        free_spaces = [(0, 0, stock_l, stock_w)]
        
        # Greedy construction with Tabu-influence
        random.shuffle(pieces)
        for p in pieces:
            if iteration < tabu_list.get(p['id'], 0):
                continue
            if tools['used_count'](current_placements, p['id']) < tools['piece_demand_max'](p['id']):
                # Find best fit among free spaces (Best Fit)
                best_space_idx = -1
                for idx, (fx, fy, fl, fw) in enumerate(free_spaces):
                    if p['l'] <= fl and p['w'] <= fw:
                        best_space_idx = idx
                        break
                
                if best_space_idx != -1:
                    fx, fy, fl, fw = free_spaces.pop(best_space_idx)
                    # Place and split free space (Guillotine cut)
                    current_placements.append((p['id'], fx, fy, p['l'], p['w'], 0))
                    
                    # Add remaining regions back to free_spaces
                    # Split along the longer dimension
                    if (fl - p['l']) > (fw - p['w']):
                        free_spaces.append((fx + p['l'], fy, fl - p['l'], fw))
                        free_spaces.append((fx, fy + p['w'], p['l'], fw - p['w']))
                    else:
                        free_spaces.append((fx, fy + p['w'], fl, fw - p['w']))
                        free_spaces.append((fx + p['l'], fy, fl - p['l'], p['w']))
        
        val = tools['total_value_of'](current_placements)
        if val > best_sol["total_value"]:
            # Verify feasibility via tools (the only expensive call)
            if tools['is_guillotine_layout'](current_placements):
                best_sol = {"total_value": val, "placements": current_placements}
                # Tabu update: penalize successful pieces
                tabu_list[random.choice(pieces)['id']] = iteration + random.randint(1, 5)

    if not best_sol["placements"]:
        fallback = tools['bottom_left_pack_demand_aware']()
        return {"total_value": tools['total_value_of'](fallback), "placements": fallback}
        
    return best_sol