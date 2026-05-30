# MACE evolved heuristic 02/10 for problem: travelling_salesman_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized hybrid heuristic combining high-quality construction (Multi-start 
    Nearest Neighbor) with aggressive local search (2-opt + Or-opt) and 
    controlled perturbation (Iterated Local Search).
    """
    start_time = time.time()
    nodes = instance['nodes']
    num_nodes = len(nodes)

    if num_nodes <= 1:
        return {'tour': list(range(num_nodes))}

    best_tour = None
    best_cost = float('inf')

    # Strategy:
    # 1. Start with a greedy NN tour to get a good baseline.
    # 2. Use a loop that alternates between:
    #    - Perturbation (random segment reversal): Escapes local optima better than random restarts.
    #    - Intensification (2-opt followed by Or-opt): Exploits the local neighborhood.
    # 3. Use time-budgeting to ensure we don't exceed the limit.

    # Initial construction
    current_tour = tools['nn_construct'](start_node=0)
    
    # Refine initial
    time_remaining = time_limit_s - (time.time() - start_time)
    if time_remaining > 0.1:
        current_tour = tools['apply_2opt'](current_tour, time_limit_s=time_remaining * 0.3)
    
    best_tour = list(current_tour)
    best_cost = tools['tour_length'](best_tour)

    # Main ILS Loop
    while time.time() - start_time < time_limit_s * 0.95:
        # Perturbation
        perturbed_tour = list(best_tour)
        if num_nodes > 4:
            # Random bridge move: swap two edges to rearrange the tour significantly
            idx = sorted(random.sample(range(num_nodes), 2))
            perturbed_tour[idx[0]:idx[1]] = reversed(perturbed_tour[idx[0]:idx[1]])
        else:
            # For tiny instances, just use a random tour to diversify
            perturbed_tour = tools['random_tour']()

        # Local Search
        remaining = time_limit_s - (time.time() - start_time)
        if remaining <= 0.05:
            break

        # Apply 2-opt for edge optimization
        current_tour = tools['apply_2opt'](
            perturbed_tour, 
            time_limit_s=remaining * 0.6, 
            first_improvement=True
        )
        
        # Apply Or-opt for node placement optimization
        current_tour = tools['apply_or_opt_single'](
            current_tour, 
            time_limit_s=max(0.05, time_limit_s - (time.time() - start_time))
        )

        # Acceptance Criterion
        current_cost = tools['tour_length'](current_tour)
        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = list(current_tour)
        
        # Periodically inject a new greedy start if we haven't improved in a while
        if random.random() < 0.1:
            current_tour = tools['nn_construct'](start_node=random.randint(0, num_nodes - 1))
            best_cost_new = tools['tour_length'](current_tour)
            if best_cost_new < best_cost:
                best_cost = best_cost_new
                best_tour = list(current_tour)

    return {'tour': best_tour}