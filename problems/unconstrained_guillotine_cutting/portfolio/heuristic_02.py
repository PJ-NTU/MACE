# MACE evolved heuristic 02/10 for problem: unconstrained_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the unconstrained guillotine cutting problem using a Greedy Randomized
    Adaptive Search Procedure (GRASP) combined with local improvement.
    """
    start_time = time.time()
    
    # Extract instance details
    pieces = instance["pieces"]
    allow_rotation = instance.get("allow_rotation", False)
    stock_w = instance["stock_width"]
    stock_h = instance["stock_height"]
    
    best_solution = {"placements": []}
    best_value = 0.0
    
    # 1. Start with the provided greedy heuristic as a baseline
    try:
        greedy_sol = tools['guillotine_greedy_value_density'](time_limit_s=0.1)
        best_solution = {"placements": greedy_sol}
        best_value = sum(pieces[p['piece_id']]['value'] for p in greedy_sol)
    except:
        pass

    # 2. Iterative improvement loop
    while time.time() - start_time < time_limit_s * 0.8:
        # Generate a candidate using a randomized greedy approach
        # Sort pieces by value density
        piece_list = list(pieces.items())
        # Shuffle slightly to introduce randomness for GRASP
        random.shuffle(piece_list)
        piece_list.sort(key=lambda x: x[1]['value'] / (x[1]['l'] * x[1]['w']), reverse=True)
        
        current_placements = []
        occupied = [] # List of (x, y, w, h)
        
        for pid, pdata in piece_list:
            # Try to find a place for this piece
            # Simple shelf-like placement strategy
            orientations = [0, 1] if allow_rotation else [0]
            
            placed = False
            for y in range(0, stock_h, 5): # Grid-based search
                for x in range(0, stock_w, 5):
                    for orient in orientations:
                        pw = pdata['w'] if orient == 1 else pdata['l']
                        ph = pdata['l'] if orient == 1 else pdata['w']
                        
                        if x + pw <= stock_w and y + ph <= stock_h:
                            # Check overlap
                            collision = False
                            for (ox, oy, ow, oh) in occupied:
                                if not (x + pw <= ox or x >= ox + ow or y + ph <= oy or y >= oy + oh):
                                    collision = True
                                    break
                            
                            if not collision:
                                current_placements.append({
                                    "piece_id": pid,
                                    "x": x,
                                    "y": y,
                                    "orientation": orient
                                })
                                occupied.append((x, y, pw, ph))
                                placed = True
                                break
                    if placed: break
                if placed: break
        
        # 3. Apply local search improvement (swap)
        try:
            improved = tools['apply_swap_local'](current_placements, t_limit=0.1)
            current_value = sum(pieces[p['piece_id']]['value'] for p in improved)
            
            if current_value > best_value:
                best_value = current_value
                best_solution = {"placements": improved}
        except:
            pass
            
    return best_solution