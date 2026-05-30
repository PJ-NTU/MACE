# MACE evolved heuristic 07/10 for problem: travelling_salesman_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves TSP using an Iterated Local Search (ILS) strategy.
    Modified the construction phase to use a Multi-Start approach with 
    Nearest Neighbor (NN) starting from multiple random nodes to find a 
    better initial basin of attraction.
    """
    start_time = time.time()
    nodes = instance['nodes']
    num_nodes = len(nodes)
    
    if num_nodes <= 1:
        return {'tour': list(range(num_nodes))}

    # Improved Construction: Multi-start NN to find a better initial candidate
    best_tour = None
    best_cost = float('inf')
    
    # Use 10% of time for multi-start construction
    construction_limit = time_limit_s * 0.1
    while time.time() - start_time < construction_limit:
        start_node = random.randint(0, num_nodes - 1)
        candidate = tools['nn_construct'](start_node=start_node)
        candidate = tools['apply_2opt'](candidate, time_limit_s=0.1, first_improvement=True)
        cost = tools['tour_length'](candidate)
        if cost < best_cost:
            best_cost = cost
            best_tour = list(candidate)
        if num_nodes < 20: break # Small instances don't need many restarts

    # Iterated Local Search (ILS) loop
    while time.time() - start_time < time_limit_s * 0.9:
        # Perturbation: apply a random 3-bridge move or segment swap to escape local optima
        perturbed_tour = list(best_tour)
        if num_nodes > 4:
            idx1, idx2 = sorted(random.sample(range(num_nodes), 2))
            # Shake: reverse a segment to diversify
            perturbed_tour[idx1:idx2] = reversed(perturbed_tour[idx1:idx2])

        # Refinement
        remaining_time = time_limit_s - (time.time() - start_time)
        if remaining_time <= 0.1:
            break
            
        current_tour = tools['apply_2opt'](
            perturbed_tour, 
            time_limit_s=max(0.05, remaining_time * 0.5), 
            first_improvement=True
        )
        
        current_tour = tools['apply_or_opt_single'](
            current_tour, 
            time_limit_s=max(0.05, remaining_time * 0.5)
        )

        # Acceptance criterion: keep if better
        current_cost = tools['tour_length'](current_tour)
        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = list(current_tour)

    return {'tour': best_tour}