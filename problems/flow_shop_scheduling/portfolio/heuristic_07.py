# MACE evolved heuristic 07/10 for problem: flow_shop_scheduling
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Simulated Annealing (SA) metaheuristic with a cooling schedule.
    
    Distinctions from portfolio:
    1. Stochastic Acceptance: Unlike the portfolio's strict hill-climbing 
       (first/best improvement), SA accepts worse moves with a probability 
       based on a temperature parameter, allowing it to escape larger 
       basins of attraction that random-swap perturbations might fail to exit.
    2. Neighborhood: Uses 'Shift' (moving one job to a new arbitrary position) 
       rather than Swap or full Insertion Search.
    3. Cooling Schedule: Implements a geometric cooling schedule to 
       transition from exploration (high T) to exploitation (low T).
    """
    start_time = time.time()
    n = instance['n']
    
    # 1. Initialization: Start with random permutation to avoid NEH bias
    # If the portfolio is too focused on NEH, this approach explores 
    # different regions of the search space.
    current_perm = list(range(1, n + 1))
    random.shuffle(current_perm)
    
    best_perm = list(current_perm)
    current_makespan = tools['simulate_makespan'](current_perm)
    best_makespan = current_makespan
    
    # 2. Simulated Annealing Parameters
    temp = 100.0
    cooling_rate = 0.9995
    min_temp = 0.01
    
    # 3. Main Loop
    # We use a shift move (pull one job index and re-insert elsewhere)
    # which is distinct from the swap/insertion-search used in the portfolio.
    while time.time() - start_time < time_limit_s * 0.95 and temp > min_temp:
        # Create a candidate by moving one job
        idx_from = random.randint(0, n - 1)
        idx_to = random.randint(0, n - 1)
        
        neighbor = list(current_perm)
        job = neighbor.pop(idx_from)
        neighbor.insert(idx_to, job)
        
        neighbor_makespan = tools['simulate_makespan'](neighbor)
        delta = neighbor_makespan - current_makespan
        
        # Acceptance criteria
        if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
            current_perm = neighbor
            current_makespan = neighbor_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_perm = list(current_perm)
        
        # Cool down
        temp *= cooling_rate
        
    return tools['make_solution'](best_perm)