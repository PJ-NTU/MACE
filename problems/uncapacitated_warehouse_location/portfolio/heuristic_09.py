# MACE evolved heuristic 09/10 for problem: uncapacitated_warehouse_location
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Core Logic: Randomized Greedy Multi-Start (GRASP-lite) combined with 
    simulated annealing-inspired perturbation.
    
    The portfolio relies heavily on deterministic local search (swap) and 
    ILP. This implementation avoids pure deterministic greedy construction 
    and instead uses a randomized constructive approach followed by 
    stochastic hill climbing to escape local optima that deterministic 
    swaps might get stuck in.
    """
    start_time = time.time()
    m = tools['n_warehouses']()
    n = tools['n_customers']()
    
    def get_time_left():
        return max(0.0, time_limit_s - (time.time() - start_time))

    best_open_set = None
    best_cost = float('inf')

    # 1. Randomized Constructive Phase (GRASP-like)
    # Instead of greedy_add/drop, we sample subsets of varying sizes.
    # We bias the sampling towards warehouses that have lower average assignment costs.
    avg_costs = []
    for i in range(m):
        c = sum(instance['customers'][j]['costs'][i] for j in range(n)) / n
        avg_costs.append(c + instance['warehouses'][i]['fixed_cost'] / (n / m + 1))
    
    # Sort indices by heuristic score
    sorted_indices = sorted(range(m), key=lambda i: avg_costs[i])
    
    # Run multiple random-start iterations
    iters = 0
    while get_time_left() > 0.1 and iters < 50:
        # Pick a random size for the open set
        k = random.randint(1, max(1, m // 2))
        # Pick k warehouses with a probability bias towards the 'best' ones
        # Using a simple tournament-style or weighted selection
        current_set = sorted(random.sample(range(m), k))
        
        # 2. Stochastic Local search: Hill Climbing with restarts
        current_cost = tools['cost_given_open'](current_set)
        
        # Hill climbing: Try single bit-flips
        improved = True
        while improved and get_time_left() > 0.05:
            improved = False
            # Shuffle indices to explore randomly
            indices = list(range(m))
            random.shuffle(indices)
            
            for i in indices:
                new_set = list(current_set)
                if i in new_set:
                    if len(new_set) > 1:
                        new_set.remove(i)
                else:
                    new_set.append(i)
                new_set.sort()
                
                new_cost = tools['cost_given_open'](new_set)
                if new_cost < current_cost:
                    current_cost = new_cost
                    current_set = new_set
                    improved = True
                    break
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_open_set = current_set
        
        iters += 1

    # 3. Final Construction
    if best_open_set is None:
        best_open_set = [0]
        
    return tools['solution_from_open'](best_open_set)