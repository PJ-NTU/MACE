# MACE evolved heuristic 10/10 for problem: packing_unequal_circles_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Population-Based Metaheuristic with Weighted Mutation.
    
    Modified: Mutation now uses a probability distribution weighted by circle size 
    (radius^2) to decide which circles to remove. This favors keeping large, high-value 
    circles while clearing smaller ones to create space, rather than uniform random 
    removal which often discards the most valuable pieces of the solution.
    """
    start_time = time.time()
    n = tools['num_circles']()
    radii = instance['radii']
    
    # Initialize population with different construction strategies
    pop_size = min(10, n + 2)
    population = []
    
    # Fill population with diverse greedy starts
    for i in range(pop_size):
        seed = random.randint(0, 1000000)
        # Mix of area-descending and random-order greedy
        if i % 2 == 0:
            coords = tools['greedy_by_area_first'](attempts_per_circle=100, rng_seed=seed)
        else:
            indices = list(range(n))
            random.shuffle(indices)
            coords = tools['greedy_pack_in_order'](indices, attempts_per_circle=100, rng_seed=seed)
        population.append(coords)
    
    best_coords = max(population, key=lambda c: tools['total_area'](c))
    best_score = tools['total_area'](best_coords)
    
    while time.time() - start_time < time_limit_s * 0.9:
        # Selection: Tournament
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        
        # Crossover: Combine subsets
        child = list(parent1)
        p2_placed = [i for i in range(n) if tools['is_placed'](i, parent2)]
        random.shuffle(p2_placed)
        
        for i in p2_placed:
            if not tools['is_placed'](i, child):
                # Try to add circle from parent 2 into parent 1's layout
                new_c = tools['try_add_circle'](child, i, attempts=50)
                if new_c:
                    child = new_c
        
        # Weighted Mutation: Clear circles, but bias towards removing smaller ones
        if random.random() < 0.3:
            placed = [i for i in range(n) if tools['is_placed'](i, child)]
            if placed:
                # Weights: 1 / (r^2 + epsilon) ensures smaller circles have higher removal probability
                weights = [1.0 / (radii[i]**2 + 1e-9) for i in placed]
                num_to_remove = max(1, len(placed) // 3)
                to_remove = random.choices(placed, weights=weights, k=num_to_remove)
                for i in set(to_remove):
                    child = tools['unplace_circle'](child, i)
        
        # Hill-climbing on child
        for _ in range(5):
            unpacked = [i for i in range(n) if not tools['is_placed'](i, child)]
            if unpacked:
                target = random.choice(unpacked)
                res = tools['try_add_circle'](child, target, attempts=100)
                if res: child = res
        
        # Update population
        population.append(child)
        population.sort(key=lambda c: tools['total_area'](c), reverse=True)
        population = population[:pop_size]
        
        # Update global best
        current_best = population[0]
        curr_score = tools['total_area'](current_best)
        if curr_score > best_score:
            best_score = curr_score
            best_coords = current_best
            
    return {"coords": best_coords}