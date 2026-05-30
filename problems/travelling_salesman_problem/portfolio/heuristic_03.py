# MACE evolved heuristic 03/10 for problem: travelling_salesman_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves TSP using an Iterated Local Search (ILS) strategy.
    Modification: Improved initialization by sampling multiple Nearest Neighbor 
    starts to find a better basin of attraction before refinement.
    """
    start_time = time.time()
    nodes = instance['nodes']
    num_nodes = len(nodes)
    
    if num_nodes <= 1:
        return {'tour': list(range(num_nodes))}

    # Initial construction: Multi-start NN to find a stronger seed
    best_tour = None
    best_cost = float('inf')
    
    # Use up to 10% of time for multi-start NN initialization
    init_time_limit = time_limit_s * 0.1
    while time.time() - start_time < init_time_limit:
        start_node = random.randint(0, num_nodes - 1)
        candidate_tour = tools['nn_construct'](start_node=start_node)
        candidate_cost = tools['tour_length'](candidate_tour)
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_tour = list(candidate_tour)
            
    # Initial refinement
    best_tour = tools['apply_2opt'](best_tour, time_limit_s=time_limit_s * 0.1, first_improvement=True)
    best_cost = tools['tour_length'](best_tour)

    # Iterated Local Search (ILS) loop
    while time.time() - start_time < time_limit_s * 0.9:
        perturbed_tour = list(best_tour)
        if num_nodes > 4:
            idx1, idx2 = sorted(random.sample(range(num_nodes), 2))
            perturbed_tour[idx1:idx2] = reversed(perturbed_tour[idx1:idx2])

        remaining_time = time_limit_s - (time.time() - start_time)
        if remaining_time <= 0.05:
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

        current_cost = tools['tour_length'](current_tour)
        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = list(current_tour)

    return {'tour': best_tour}