# MACE evolved heuristic 09/10 for problem: travelling_salesman_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves TSP using an Iterated Local Search (ILS) strategy.
    Modification: Enhanced the perturbation phase by using a 'double-bridge' 
    move instead of a simple segment reversal to better escape deep local 
    minima in larger instances.
    """
    start_time = time.time()
    nodes = instance['nodes']
    num_nodes = len(nodes)
    
    if num_nodes <= 1:
        return {'tour': list(range(num_nodes))}

    best_tour = None
    best_cost = float('inf')
    
    construction_limit = time_limit_s * 0.1
    while time.time() - start_time < construction_limit:
        start_node = random.randint(0, num_nodes - 1)
        candidate = tools['nn_construct'](start_node=start_node)
        candidate = tools['apply_2opt'](candidate, time_limit_s=0.1, first_improvement=True)
        cost = tools['tour_length'](candidate)
        if cost < best_cost:
            best_cost = cost
            best_tour = list(candidate)
        if num_nodes < 20: break

    while time.time() - start_time < time_limit_s * 0.9:
        perturbed_tour = list(best_tour)
        if num_nodes > 8:
            # Double-bridge move: 4-change that is highly effective at scrambling 
            # the tour while maintaining structure better than random reversals.
            p = sorted(random.sample(range(num_nodes), 4))
            a, b, c, d = p[0], p[1], p[2], p[3]
            perturbed_tour = (perturbed_tour[:a+1] + 
                              perturbed_tour[c+1:d+1] + 
                              perturbed_tour[b+1:c+1] + 
                              perturbed_tour[a+1:b+1] + 
                              perturbed_tour[d+1:])

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

        current_cost = tools['tour_length'](current_tour)
        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = list(current_tour)

    return {'tour': best_tour}