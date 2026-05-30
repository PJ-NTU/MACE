# MACE evolved heuristic 08/10 for problem: packing_unequal_circles_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Hybrid heuristic combining multi-start greedy construction with a
    Metropolis-Hastings local search.
    
    1. Construction: Uses randomized greedy permutations to explore diverse
       packing topologies, overcoming h_b's deterministic bias.
    2. Local Search: Employs an SA-like acceptance criterion to allow 
       temporary area losses, effectively escaping the local optima that 
       h_b gets trapped in.
    3. Refinement: Prioritizes high-value swaps between smallest-packed and
       largest-unpacked circles, while maintaining structural integrity 
       through random relocations.
    """
    start_time = time.time()
    n = tools['num_circles']()
    radii = instance['radii']
    
    # Pre-calculate indices sorted by area for efficiency
    indices_by_area = sorted(range(n), key=lambda i: radii[i], reverse=True)
    
    best_coords = tools['unpacked_template']()
    best_area = 0.0
    
    # SA Parameters
    temp = 1.0
    cooling_rate = 0.9999
    
    # Main loop
    while time.time() - start_time < time_limit_s * 0.9:
        # Multi-start: Alternating between pure greedy and randomized greedy
        if random.random() < 0.3:
            # Fully randomized order
            order = list(range(n))
            random.shuffle(order)
            curr_coords = tools['greedy_pack_in_order'](order, attempts_per_circle=100)
        else:
            # Partially randomized greedy order (top-heavy)
            order = indices_by_area[:]
            # Shuffle the bottom 50%
            split = n // 2
            random.shuffle(order[split:])
            curr_coords = tools['greedy_pack_in_order'](order, attempts_per_circle=100)
            
        curr_area = tools['total_area'](curr_coords)
        
        # Local Search Phase (Simulated Annealing)
        for _ in range(100):
            if time.time() - start_time > time_limit_s * 0.95:
                break
                
            move = random.random()
            next_coords = None
            
            # Neighborhood operators
            if move < 0.4:  # Try to add
                unpacked = [i for i in range(n) if not tools['is_placed'](i, curr_coords)]
                if unpacked:
                    target = max(unpacked, key=lambda i: radii[i])
                    next_coords = tools['try_add_circle'](curr_coords, target, attempts=100)
            elif move < 0.7:  # Try to swap
                placed = [i for i in range(n) if tools['is_placed'](i, curr_coords)]
                unpacked = [i for i in range(n) if not tools['is_placed'](i, curr_coords)]
                if placed and unpacked:
                    out_i = min(placed, key=lambda i: radii[i])
                    in_i = max(unpacked, key=lambda i: radii[i])
                    if radii[in_i] > radii[out_i]:
                        next_coords = tools['try_swap_in_out'](curr_coords, out_i, in_i, attempts=100)
            else:  # Relocate
                placed = [i for i in range(n) if tools['is_placed'](i, curr_coords)]
                if placed:
                    next_coords = tools['try_relocate_circle'](curr_coords, random.choice(placed), attempts=50)
            
            if next_coords:
                next_area = tools['total_area'](next_coords)
                delta = next_area - curr_area
                
                # Metropolis acceptance
                if delta > 0 or (temp > 0 and random.random() < math.exp(delta / (temp * 0.1))):
                    curr_coords = next_coords
                    curr_area = next_area
                    if curr_area > best_area:
                        best_area = curr_area
                        best_coords = list(curr_coords)
                        
            temp *= cooling_rate
            
    return {"coords": best_coords}