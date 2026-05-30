# MACE evolved heuristic 06/10 for problem: travelling_salesman_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined TSP Heuristic:
    - Adaptive Search: Uses size-based strategy (Multi-start vs ILS).
    - Randomized Construction: Uses varied Nearest Neighbor seeds to diversify.
    - Robust Local Search: 2-opt + Or-opt sequence with strict time management.
    - Perturbation: Uses randomized segment reversal rather than complex 4-opt 
      which can be overly disruptive on smaller instances.
    """
    start_time = time.time()
    nodes = instance['nodes']
    num_nodes = len(nodes)

    if num_nodes <= 1:
        return {'tour': list(range(num_nodes))}
    
    # Pre-calculate time budget for initial vs refinement
    deadline = start_time + time_limit_s * 0.95
    
    best_tour = None
    best_cost = float('inf')

    # Strategy: Multi-start with local search provides better coverage for 
    # small-to-medium instances, while maintaining a single best for large ones.
    
    # 1. Quick initial construction
    current_tour = tools['nn_construct'](start_node=0)
    best_tour = tools['apply_2opt'](current_tour, time_limit_s=time_limit_s * 0.1)
    best_cost = tools['tour_length'](best_tour)

    # 2. Iterative optimization loop
    iteration = 0
    while time.time() < deadline:
        # Diversification: Every 5 iterations or if time allows, restart with new seed
        if iteration % 5 == 0:
            start_node = random.randint(0, num_nodes - 1)
            candidate = tools['nn_construct'](start_node=start_node)
        else:
            # Perturbation: Random segment reversal
            candidate = list(best_tour)
            if num_nodes > 2:
                a, b = sorted(random.sample(range(num_nodes), 2))
                candidate[a:b] = reversed(candidate[a:b])
        
        # Local Search
        remaining = deadline - time.time()
        if remaining <= 0.01:
            break
            
        candidate = tools['apply_2opt'](candidate, time_limit_s=remaining * 0.5)
        candidate = tools['apply_or_opt_single'](candidate, time_limit_s=remaining * 0.5)
        
        # Evaluation
        cost = tools['tour_length'](candidate)
        if cost < best_cost:
            best_cost = cost
            best_tour = candidate
            
        iteration += 1

    return {'tour': best_tour}