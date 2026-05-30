# MACE evolved heuristic 08/10 for problem: travelling_salesman_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style TSP solver.
    
    Hypothesis:
    - Smaller instances (N < 200) benefit from high-diversity multi-start (Parent B)
      because they are more prone to getting stuck in local minima of 2-opt.
    - Larger instances (N >= 200) benefit from Iterated Local Search (Parent A)
      which maintains a strong 'incumbent' and performs incremental perturbations
      to explore the search space efficiently without excessive restarts.
    """
    start_time = time.time()
    nodes = instance['nodes']
    num_nodes = len(nodes)

    if num_nodes <= 1:
        return {'tour': list(range(num_nodes))}

    # Dispatcher strategy based on problem scale
    is_large_instance = num_nodes >= 200

    if not is_large_instance:
        # Parent B strategy: Multi-start focus
        deadline = start_time + time_limit_s * 0.96
        best_tour = tools['nn_construct'](start_node=0)
        best_tour = tools['apply_2opt'](best_tour, time_limit_s=time_limit_s * 0.1)
        best_cost = tools['tour_length'](best_tour)

        iterations = 0
        while time.time() < deadline:
            if iterations % 4 == 0:
                candidate = tools['nn_construct'](start_node=random.randint(0, num_nodes - 1))
            else:
                candidate = list(best_tour)
                if num_nodes > 3:
                    a, b = sorted(random.sample(range(num_nodes), 2))
                    candidate[a:b] = reversed(candidate[a:b])
            
            rem = deadline - time.time()
            if rem < 0.01: break
            
            candidate = tools['apply_2opt'](candidate, time_limit_s=rem * 0.6)
            candidate = tools['apply_or_opt_single'](candidate, time_limit_s=max(0.01, deadline - time.time()))
            
            cost = tools['tour_length'](candidate)
            if cost < best_cost:
                best_cost, best_tour = cost, candidate
            iterations += 1
    else:
        # Parent A strategy: ILS focus
        current_tour = tools['nn_construct'](start_node=0)
        best_tour = tools['apply_2opt'](current_tour, time_limit_s=time_limit_s * 0.2)
        best_cost = tools['tour_length'](best_tour)

        while time.time() - start_time < time_limit_s * 0.95:
            perturbed = list(best_tour)
            # Use a larger perturbation for large instances
            if num_nodes > 10:
                a, b = sorted(random.sample(range(num_nodes), 2))
                perturbed[a:b] = reversed(perturbed[a:b])
            
            rem = time_limit_s - (time.time() - start_time)
            if rem < 0.05: break
            
            current = tools['apply_2opt'](perturbed, time_limit_s=rem * 0.5, first_improvement=True)
            current = tools['apply_or_opt_single'](current, time_limit_s=max(0.02, time_limit_s - (time.time() - start_time)))
            
            cost = tools['tour_length'](current)
            if cost < best_cost:
                best_cost, best_tour = cost, current

    return {'tour': best_tour}