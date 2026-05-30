# MACE evolved heuristic 09/10 for problem: euclidean_steiner_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A population-based Evolutionary Strategy (ES) heuristic for the Euclidean Steiner Problem.
    
    Departure from portfolio:
    - Population-based: Instead of a single path/trajectory (SA or hill-climbing), maintain 
      a population of Steiner point sets and evolve them using crossover and mutation.
    - Crossover: Combines two parent solutions by mixing their Steiner point sets,
      a technique missing from the trajectory-based portfolio.
    - No direct coordinate descent until final stage: Focuses on global search via
      evolutionary operators to avoid premature local convergence.
    """
    start_time = time.time()
    points = instance.get("points", [])
    if not points:
        return {"steiner_points": []}

    # 1. Initialization: Create a diverse population
    pop_size = 8
    population = []
    for _ in range(pop_size):
        # Seed with different levels of greedy Fermat points
        s = tools['add_fermat_points_for_mst_triples'](min_improvement=random.uniform(1e-10, 1e-6))
        # Add some random jitter to ensure diversity
        if random.random() < 0.5:
            s = [(p[0] + random.gauss(0, 0.1), p[1] + random.gauss(0, 0.1)) for p in s]
        population.append(s)

    # 2. Evolutionary Loop
    while time.time() - start_time < time_limit_s * 0.8:
        # Score population
        scores = [tools['mst_length'](p) for p in population]
        
        # Selection: Tournament
        idx1, idx2 = random.sample(range(pop_size), 2)
        parent1 = population[idx1] if scores[idx1] < scores[idx2] else population[idx2]
        
        idx3, idx4 = random.sample(range(pop_size), 2)
        parent2 = population[idx3] if scores[idx3] < scores[idx4] else population[idx4]
        
        # Crossover: Uniform recombination of sets
        child = []
        combined = list(set(parent1 + parent2))
        for p in combined:
            if random.random() < 0.5:
                child.append(p)
        
        # Mutation: Add or shift
        if random.random() < 0.3 and child:
            idx = random.randrange(len(child))
            child[idx] = (child[idx][0] + random.gauss(0, 0.05), child[idx][1] + random.gauss(0, 0.05))
        elif random.random() < 0.2:
            idx = random.randrange(len(points))
            child.append(points[idx])
            
        # Replacement
        worst_idx = scores.index(max(scores))
        population[worst_idx] = child

    # 3. Final Polish: Find the best in population and refine
    final_scores = [tools['mst_length'](p) for p in population]
    best_steiner = population[final_scores.index(min(final_scores))]
    
    refined = tools['local_relocate_steiner'](
        steiner_points=best_steiner,
        time_limit_s=max(0.05, time_limit_s - (time.time() - start_time)),
        step=0.01
    )
    
    return tools['make_solution'](steiner_points=refined)