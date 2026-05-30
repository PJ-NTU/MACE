# MACE evolved heuristic 05/10 for problem: assortment_problem
import time
import random
import math
import itertools

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Core Logic: Population-based evolutionary search (Genetic Algorithm style).
    - Portfolio gap: Most portfolio members are single-point local search / hill-climbers.
    - This heuristic maintains a population of diverse, feasible solutions and 
      applies crossover-like recombination (swapping stock instances between solutions) 
      and mutation (randomizing piece counts).
    """
    start_time = time.time()
    
    # 1. Initialize Population
    population = []
    n_stocks = tools['n_stocks']()
    m = tools['n_types']()
    
    # Generate initial diverse population
    for _ in range(5):
        st = random.randint(1, n_stocks)
        sol = tools['greedy_for_bounds'](stock_type=st, prefer='max')
        if sol:
            population.append(sol)
            
    if not population:
        population.append(tools['solve_default'](time_limit_s=0.5))

    best_sol = min(population, key=lambda x: tools['objective'](x))
    best_obj = tools['objective'](best_sol)

    # 2. Evolutionary Loop
    while time.time() - start_time < time_limit_s * 0.9:
        # Selection: Pick two parents
        if len(population) < 2:
            parent1 = population[0]
            parent2 = tools['solve_default'](time_limit_s=0.1)
        else:
            parent1, parent2 = random.sample(population, 2)
        
        # Recombination: Merge stock instances from two solutions
        # (Only if they share stock types or use <= 2 types)
        new_placements = {**parent1['placements']}
        # Inject some instances from parent2
        for k, v in parent2['placements'].items():
            new_placements[len(new_placements) + 1] = v
        
        # Trim to valid stock types (must be <= 2)
        used_types = {inst['stock_type'] for inst in new_placements.values()}
        if len(used_types) > 2:
            # Drop random until valid
            while len(used_types) > 2:
                rem = random.choice(list(new_placements.keys()))
                del new_placements[rem]
                used_types = {inst['stock_type'] for inst in new_placements.values()}
        
        child = {'placements': new_placements}
        
        # Mutation: Try to add/remove pieces using existing apply_swap_pieces
        # or simple local adjustment
        for _ in range(3):
            t1, t2 = random.randint(1, m), random.randint(1, m)
            child = tools['apply_swap_pieces'](child, t1, t2) or child
            
        # Feasibility Check
        is_ok, _ = tools['is_feasible'](child)
        if is_ok:
            child_obj = tools['objective'](child)
            # Survival of the fittest
            if len(population) < 10:
                population.append(child)
            else:
                worst = max(population, key=lambda x: tools['objective'](x))
                if child_obj < tools['objective'](worst):
                    population.remove(worst)
                    population.append(child)
            
            if child_obj < best_obj:
                best_obj = child_obj
                best_sol = child
        else:
            # If child is infeasible, try to repair by using greedy_minimal_feasible
            if random.random() < 0.2:
                population.append(tools['greedy_minimal_feasible']())
                
    return best_sol