# MACE evolved heuristic 09/10 for problem: flow_shop_scheduling
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Modified Simulated Annealing (SA) metaheuristic.
    
    Modification: Replaced the random initialization with the NEH heuristic 
    (the gold-standard flow-shop construction). Starting from a high-quality 
    NEH solution significantly reduces the time required to find the 
    global optimum, especially for larger problem instances.
    """
    start_time = time.time()
    n = instance['n']
    
    # 1. Initialization: Start with NEH construction for a strong warm-start
    current_perm = tools['neh_construct']()
    
    best_perm = list(current_perm)
    current_makespan = tools['simulate_makespan'](current_perm)
    best_makespan = current_makespan
    
    # 2. Simulated Annealing Parameters
    temp = 100.0
    cooling_rate = 0.9995
    min_temp = 0.01
    
    # 3. Main Loop
    # We use a shift move (pull one job index and re-insert elsewhere)
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