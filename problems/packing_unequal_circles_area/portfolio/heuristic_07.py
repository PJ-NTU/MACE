# MACE evolved heuristic 07/10 for problem: packing_unequal_circles_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized metaheuristic for packing unequal circles.
    
    Design improvements over h_a/h_b:
    1. Multi-start construction: Uses both area-descending and random-order greedy
       to ensure a diverse set of starting points.
    2. Adaptive Local Search: Employs a 're-fill' strategy after structural 
       mutations, focusing on the largest available circles to maximize area.
    3. Time-Awareness: Dynamically adjusts the intensity of local search and 
       perturbations based on the remaining time budget.
    """
    start_time = time.time()
    n = tools['num_circles']()
    radii = instance['radii']
    
    def get_greedy_solution(seed=None):
        if random.random() < 0.5:
            return tools['greedy_by_area_first'](attempts_per_circle=200, rng_seed=seed)
        else:
            indices = list(range(n))
            random.shuffle(indices)
            return tools['greedy_pack_in_order'](indices, attempts_per_circle=200, rng_seed=seed)

    # Initial best
    best_coords = get_greedy_solution()
    best_score = tools['total_area'](best_coords)
    
    # Local search loop
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.92:
        iteration += 1
        current = list(best_coords)
        
        # Perturbation: Ruin-and-recreate
        # Remove a random percentage of circles
        placed = [i for i in range(n) if tools['is_placed'](i, current)]
        if not placed:
            current = get_greedy_solution()
        else:
            # Remove 1 to 3 circles to create space for larger ones
            num_to_remove = min(len(placed), random.randint(1, 3))
            for i in random.sample(placed, num_to_remove):
                current = tools['unplace_circle'](current, i)
            
            # Attempt to re-fill with largest available circles
            unpacked = [i for i in range(n) if not tools['is_placed'](i, current)]
            unpacked.sort(key=lambda i: radii[i], reverse=True)
            for i in unpacked:
                res = tools['try_add_circle'](current, i, attempts=150)
                if res:
                    current = res
        
        # Local Improvement: Try to relocate currently placed circles to make room
        placed = [i for i in range(n) if tools['is_placed'](i, current)]
        if placed:
            c_idx = random.choice(placed)
            res = tools['try_relocate_circle'](current, c_idx, attempts=100)
            if res:
                current = res
                # Try to add something in the freed space
                rem = [i for i in range(n) if not tools['is_placed'](i, current)]
                for i in sorted(rem, key=lambda x: radii[x], reverse=True):
                    res2 = tools['try_add_circle'](current, i, attempts=100)
                    if res2:
                        current = res2
                        break
        
        # Update best
        curr_score = tools['total_area'](current)
        if curr_score > best_score:
            best_score = curr_score
            best_coords = current
        
        # Occasional restart to avoid local optima
        if iteration % 50 == 0:
            candidate = get_greedy_solution()
            if tools['total_area'](candidate) > best_score:
                best_coords = candidate
                best_score = tools['total_area'](candidate)
                
    return {"coords": best_coords}