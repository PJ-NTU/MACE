# MACE evolved heuristic 01/10 for problem: travelling_salesman_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves TSP using an Iterated Local Search (ILS) strategy.
    Improves upon the MSLS by using a 'perturbation' mechanism instead of purely 
    random restarts, which helps explore the neighborhood of promising local optima.
    """
    start_time = time.time()
    nodes = instance['nodes']
    num_nodes = len(nodes)
    
    if num_nodes <= 1:
        return {'tour': list(range(num_nodes))}

    # Initial construction
    current_tour = tools['nn_construct'](start_node=0)
    current_tour = tools['apply_2opt'](current_tour, time_limit_s=time_limit_s * 0.2, first_improvement=True)
    
    best_tour = list(current_tour)
    best_cost = tools['tour_length'](best_tour)

    # Iterated Local Search (ILS) loop
    while time.time() - start_time < time_limit_s * 0.9:
        # Perturbation: apply a random 3-bridge move or segment swap to escape local optima
        # Instead of a full random restart, we start from the current best to refine it
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