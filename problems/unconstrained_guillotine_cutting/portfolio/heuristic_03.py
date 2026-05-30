# MACE evolved heuristic 03/10 for problem: unconstrained_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the unconstrained guillotine cutting problem using a Greedy Randomized
    Adaptive Search Procedure (GRASP) combined with local improvement.
    Modified to improve the construction phase by using a dynamic search grid 
    based on piece dimensions to fill the stock more densely.
    """
    start_time = time.time()
    
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
        piece_list = list(pieces.items())
        # Shuffle with weighted bias to favor high-density pieces while maintaining exploration
        random.shuffle(piece_list)
        piece_list.sort(key=lambda x: x[1]['value'] / (x[1]['l'] * x[1]['w']), reverse=True)
        
        current_placements = []
        occupied = [] # List of (x, y, w, h)
        
        # Track occupied space using a finer, adaptive scan
        for pid, pdata in piece_list:
            orientations = [0, 1] if allow_rotation else [0]
            
            placed = False
            # Optimization: Instead of fixed 5x5, probe available coordinates from existing piece boundaries
            candidate_points = [(0, 0)]
            for (ox, oy, ow, oh) in occupied:
                candidate_points.append((ox + ow, oy))
                candidate_points.append((ox, oy + oh))
                
            for x, y in candidate_points:
                for orient in orientations:
                    pw = pdata['w'] if orient == 1 else pdata['l']
                    ph = pdata['l'] if orient == 1 else pdata['w']
                    
                    if x + pw <= stock_w and y + ph <= stock_h:
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
        
        # 3. Apply local search improvement
        try:
            improved = tools['apply_swap_local'](current_placements, t_limit=0.1)
            current_value = sum(pieces[p['piece_id']]['value'] for p in improved)
            
            if current_value > best_value:
                best_value = current_value
                best_solution = {"placements": improved}
        except:
            pass
            
    return best_solution