# MACE evolved heuristic 04/10 for problem: travelling_salesman_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized hybrid heuristic combining high-quality construction (Multi-start 
    Nearest Neighbor) with aggressive local search (2-opt + Or-opt) and 
    controlled perturbation (Iterated Local Search).
    
    Modification: Improved the perturbation strategy from a simple segment reversal
    to a double-bridge move (4-opt equivalent), which is significantly more effective 
    at escaping local minima in TSP landscapes.
    """
    start_time = time.time()
    nodes = instance['nodes']
    num_nodes = len(nodes)

    if num_nodes <= 1:
        return {'tour': list(range(num_nodes))}

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
        # Perturbation: Double-bridge move (4-opt)
        perturbed_tour = list(best_tour)
        if num_nodes > 8:
            # Four cut points for the double bridge
            p = sorted(random.sample(range(num_nodes), 4))
            # Split into 4 segments: A, B, C, D
            # Reconnect as A-C-B-D
            A = perturbed_tour[:p[0]]
            B = perturbed_tour[p[0]:p[1]]
            C = perturbed_tour[p[1]:p[2]]
            D = perturbed_tour[p[2]:p[3]]
            E = perturbed_tour[p[3]:]
            perturbed_tour = A + C + B + D + E
        else:
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
        
        # Periodically inject a new greedy start
        if random.random() < 0.1:
            current_tour = tools['nn_construct'](start_node=random.randint(0, num_nodes - 1))
            best_cost_new = tools['tour_length'](current_tour)
            if best_cost_new < best_cost:
                best_cost = best_cost_new
                best_tour = list(current_tour)

    return {'tour': best_tour}