# MACE evolved heuristic 05/10 for problem: packing_unequal_circles_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Population-based Evolutionary heuristic (Memetic Algorithm) for packing.
    
    Departure from portfolio:
    1. Population-based: Instead of iterating on a single solution, it maintains
       a small population of diverse feasible configurations.
    2. Crossover: Implements a 'Union-Crossover' where two parents are combined
       by taking the union of their placed circles, then repairing feasibility.
    3. Mutation: Uses a structural 'shake' (randomly removing a subset of circles)
       followed by greedy re-filling, rather than simple local swaps.
    """
    start_time = time.time()
    n = tools['num_circles']()
    radii = instance['radii']
    
    # Population size
    POP_SIZE = 5
    population = []
    
    # Initialization: Create a diverse initial population
    for i in range(POP_SIZE):
        # Use different randomized greedy orderings
        indices = list(range(n))
        random.shuffle(indices)
        coords = tools['greedy_pack_in_order'](indices, attempts_per_circle=100)
        population.append(coords)
    
    best_coords = max(population, key=lambda c: tools['total_area'](c))
    best_score = tools['total_area'](best_coords)
    
    while time.time() - start_time < time_limit_s * 0.9:
        # Selection: Pick two parents
        p1, p2 = random.sample(population, 2)
        
        # Crossover: Union the placed circles
        child = tools['unpacked_template']()
        p1_placed = [i for i in range(n) if tools['is_placed'](i, p1)]
        p2_placed = [i for i in range(n) if tools['is_placed'](i, p2)]
        
        # Combine unique circles from parents, prioritize largest
        candidate_indices = sorted(list(set(p1_placed + p2_placed)), key=lambda i: radii[i], reverse=True)
        
        # Attempt to pack the union
        for i in candidate_indices:
            child = tools['try_add_circle'](child, i, attempts=100) or child
            
        # Mutation: Shake (remove random 20%) and fill
        if random.random() < 0.3:
            placed = [i for i in range(n) if tools['is_placed'](i, child)]
            if len(placed) > 2:
                for _ in range(max(1, len(placed) // 5)):
                    child = tools['unplace_circle'](child, random.choice(placed))
                # Fill gaps greedily
                rem = [i for i in range(n) if not tools['is_placed'](i, child)]
                random.shuffle(rem)
                for i in rem:
                    child = tools['try_add_circle'](child, i, attempts=50) or child
                    
        # Update population
        population.append(child)
        population.sort(key=lambda c: tools['total_area'](c), reverse=True)
        population = population[:POP_SIZE]
        
        # Update global best
        current_top = population[0]
        current_score = tools['total_area'](current_top)
        if current_score > best_score:
            best_score = current_score
            best_coords = current_top
            
    return {"coords": best_coords}