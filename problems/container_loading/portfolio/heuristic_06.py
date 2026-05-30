# MACE evolved heuristic 06/10 for problem: container_loading
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Simulated Annealing' meta-heuristic based on a 'Layered Shelf' 
    packing strategy.
    
    Departure from portfolio:
    1. Instead of extreme-point greedy packing (Corner/Wall), this uses a 
       'Shelf-Level' approach: boxes are packed in layers (shelves). 
    2. The state representation is not just a list of placements, but an 
       ordered sequence of box-types and their orientations.
    3. The search space is the permutation of all available boxes, 
       optimized via Simulated Annealing to maximize volume utilization 
       within a deterministic shelf-packer.
    """
    start_time = time.time()
    box_types = instance['box_types']
    
    # Flatten all available boxes into a list
    all_boxes = []
    for bt, info in box_types.items():
        all_boxes.extend([bt] * info['count'])
    
    # Shelf-packing implementation
    def pack_sequence(sequence):
        placements = []
        # Simple shelf: x=0, y=0, z=0
        # Tracks current shelf height and current x/y cursor
        container = tools['container_dims']()
        cw, ch, cd = container
        
        curr_x, curr_y, curr_z = 0, 0, 0
        shelf_ht = 0
        
        for bt in sequence:
            # Try all allowed orientations
            orientations = tools['box_orientations'](bt)
            for v, hswap, sx, sy, sz in orientations:
                # Check if fits in current shelf
                if curr_x + sx <= cw and curr_y + sy <= ch and curr_z + sz <= cd:
                    p = {
                        'box_type': bt, 'container_id': 0,
                        'x': curr_x, 'y': curr_y, 'z': curr_z,
                        'v': v, 'hswap': hswap
                    }
                    placements.append(p)
                    curr_x += sx
                    shelf_ht = max(shelf_ht, sz)
                    if curr_x + sx > cw:
                        curr_x = 0
                        curr_y += sy # Move to next row in shelf
                        if curr_y + sy > ch:
                            curr_y = 0
                            curr_z += shelf_ht # Move to next shelf
                            shelf_ht = 0
                    break
        return placements

    # Initial state
    random.shuffle(all_boxes)
    current_seq = list(all_boxes)
    best_placements = pack_sequence(current_seq)
    best_util = tools['utilization'](best_placements)
    
    # SA Parameters
    temp = 1.0
    cooling = 0.995
    
    while (time.time() - start_time) < (time_limit_s * 0.9):
        # Perturb: Swap two random items in the sequence
        i, j = random.sample(range(len(current_seq)), 2)
        current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
        
        new_placements = pack_sequence(current_seq)
        new_util = tools['utilization'](new_placements)
        
        # Metropolis acceptance
        if new_util > best_util or (temp > 0.001 and math.exp((new_util - best_util) / temp) > random.random()):
            if new_util > best_util:
                best_util = new_util
                best_placements = new_placements
        else:
            # Backtrack
            current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
            
        temp *= cooling

    return tools['make_solution'](best_placements)