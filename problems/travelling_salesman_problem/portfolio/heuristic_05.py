# MACE evolved heuristic 05/10 for problem: travelling_salesman_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver:
    - Small instances (N < 100): High-intensity 2-opt/Or-opt refinement (Strategy A).
    - Large instances (N >= 100): Iterated Local Search with Double-Bridge 
      perturbations to escape local optima (Strategy B).
    """
    start_time = time.time()
    nodes = instance['nodes']
    n = len(nodes)
    
    if n <= 1:
        return {'tour': list(range(n))}
    
    # Strategy A: Focused Local Search (Good for smaller N)
    def strategy_a():
        best_tour = None
        best_cost = float('inf')
        iteration = 0
        while time.time() - start_time < time_limit_s * 0.95:
            s_node = iteration % n if iteration < n else random.randint(0, n - 1)
            current_tour = tools['nn_construct'](start_node=s_node)
            
            remaining = time_limit_s - (time.time() - start_time)
            if remaining < 0.01: break
                
            current_tour = tools['apply_2opt'](current_tour, time_limit_s=remaining * 0.7, first_improvement=True)
            current_tour = tools['apply_or_opt_single'](current_tour, time_limit_s=remaining * 0.2)
            
            current_cost = tools['tour_length'](current_tour)
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = current_tour
            iteration += 1
        return best_tour or tools['nn_construct'](start_node=0)

    # Strategy B: Iterated Local Search (Good for larger N)
    def strategy_b():
        best_tour = None
        best_cost = float('inf')
        
        # Initial greedy seed
        init_deadline = start_time + (time_limit_s * 0.1)
        for i in range(min(n, 20)): # Sample starters for large instances
            if time.time() > init_deadline: break
            candidate = tools['nn_construct'](start_node=i)
            cost = tools['tour_length'](candidate)
            if cost < best_cost:
                best_cost = cost
                best_tour = candidate
        
        best_tour = tools['apply_2opt'](best_tour, time_limit_s=time_limit_s * 0.1)
        
        while time.time() - start_time < time_limit_s * 0.95:
            if n >= 8:
                pts = sorted(random.sample(range(n), 4))
                p1, p2, p3, p4 = pts
                new_tour = best_tour[0:p1] + best_tour[p3:p4] + best_tour[p2:p3] + best_tour[p1:p2] + best_tour[p4:]
            else:
                new_tour = list(best_tour)
                random.shuffle(new_tour)
                
            remaining = time_limit_s - (time.time() - start_time)
            if remaining < 0.01: break
                
            current = tools['apply_2opt'](new_tour, time_limit_s=remaining * 0.6)
            current = tools['apply_or_opt_single'](current, time_limit_s=remaining * 0.3)
            
            current_cost = tools['tour_length'](current)
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = current
        return best_tour

    # Dispatch based on complexity
    if n < 100:
        result = strategy_a()
    else:
        result = strategy_b()
        
    return {'tour': result}