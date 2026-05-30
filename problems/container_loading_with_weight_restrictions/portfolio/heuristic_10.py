# MACE evolved heuristic 10/10 for problem: container_loading_with_weight_restrictions
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Simulated Annealing' heuristic that performs stochastic hill-climbing
    on the *sequence* of boxes rather than the spatial construction.
    
    Difference from portfolio:
    1. Uses a global sequence permutation rather than shelf-based construction.
    2. Instead of building shelves, it uses a 'deep-first' placement (Backtracking-lite)
       where it tries to fit boxes into the first available void found by a 
       greedy depth-first search of the 3D volume.
    3. Employs a cooling schedule to escape local optima, unlike the greedy/random 
       multi-start approaches in the portfolio.
    """
    start_time = time.time()
    container = tools['container_dims']()
    box_types = instance['box_types']
    
    # Flatten box list for sequence optimization
    all_boxes = []
    for i, bt in enumerate(box_types):
        for _ in range(bt['count']):
            all_boxes.append(i + 1)
            
    best_placements = []
    best_util = -1.0
    
    # Current sequence
    current_seq = all_boxes[:]
    random.shuffle(current_seq)
    
    temp = 1.0
    cooling = 0.99
    
    while time.time() - start_time < time_limit_s * 0.9:
        # Generate neighbor by swapping two random elements
        if len(current_seq) > 1:
            i, j = random.sample(range(len(current_seq)), 2)
            current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
            
        # Attempt packing using the sequence
        # Strategy: Greedy fit into the first available (x, y, z) corner
        # to maximize density without shelf constraints
        placements = []
        occupied = [] # List of (x, y, z, dx, dy, dz)
        
        current_counts = {bt_idx: 0 for bt_idx in range(1, len(box_types) + 1)}
        
        for bt_idx in current_seq:
            if current_counts[bt_idx] >= box_types[bt_idx-1]['count']:
                continue
                
            # Try to place box at (0,0,0) or adjacent to existing boxes
            placed_flag = False
            # Candidates: (0,0,0) and corners of existing boxes
            candidates = [(0, 0, 0)]
            for p in occupied:
                candidates.append((p[0] + p[3], p[1], p[2]))
                candidates.append((p[0], p[1] + p[4], p[2]))
                candidates.append((p[0], p[1], p[2] + p[5]))
                
            for x, y, z in candidates:
                for orient in tools['allowed_orientations'](bt_idx):
                    dx, dy, dz = tools['box_dims'](bt_idx, orient)
                    
                    # Bounds check
                    if x + dx > container[0] or y + dy > container[1] or z + dz > container[2]:
                        continue
                    
                    # Overlap check
                    overlap = False
                    for p in occupied:
                        if not (x + dx <= p[0] or x >= p[0] + p[3] or
                                y + dy <= p[1] or y >= p[1] + p[4] or
                                z + dz <= p[2] or z >= p[2] + p[5]):
                            overlap = True
                            break
                    if overlap: continue
                    
                    # Place
                    placements.append({'box_type': bt_idx, 'orientation': orient, 'x': float(x), 'y': float(y), 'z': float(z)})
                    occupied.append((x, y, z, dx, dy, dz))
                    current_counts[bt_idx] += 1
                    placed_flag = True
                    break
                if placed_flag: break
        
        # Evaluate
        sol = tools['make_solution'](placements)
        is_ok, _ = tools['is_feasible'](sol)
        
        if is_ok:
            util = tools['utilization'](placements)
            if util > best_util:
                best_util = util
                best_placements = placements
            elif random.random() < np.exp((util - best_util) / temp):
                best_util = util
                best_placements = placements
        
        temp *= cooling
        if temp < 1e-4: break
        
    return tools['make_solution'](best_placements)