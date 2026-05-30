# MACE evolved heuristic 04/10 for problem: generalised_assignment_problem
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Simulated Annealing (SA) heuristic for GAP.
    
    The portfolio is dominated by Hill Climbing (greedy local improvement) and ILP.
    This heuristic differentiates by:
    1. Using a probabilistic acceptance criterion (Metropolis) to accept 
       worse solutions, allowing it to escape local optima that trap Hill Climbing.
    2. Employing a cooling schedule (Exponential) rather than purely greedy descent.
    3. Not relying on ILP as a core engine, focusing instead on pure stochastic 
       exploration across the entire feasible neighborhood.
    """
    start_time = time.time()
    n = instance['n']
    m = instance['m']
    
    # Construction: Randomized Greedy to provide a diverse starting point
    # rather than just the deterministic greedy_min_cost.
    def get_randomized_greedy():
        assignment = [0] * n
        rem_cap = list(instance['capacities'])
        tasks = list(range(n))
        random.shuffle(tasks)
        for j in tasks:
            feasible = [i for i in range(1, m + 1) if instance['consumption_matrix'][i-1][j] <= rem_cap[i-1]]
            if not feasible: return None
            # Pick randomly from feasible agents to ensure diversity
            choice = random.choice(feasible)
            assignment[j] = choice
            rem_cap[choice-1] -= instance['consumption_matrix'][choice-1][j]
        return assignment

    curr_assignment = get_randomized_greedy()
    if not curr_assignment:
        curr_assignment = tools['greedy_min_cost']()
        
    best_assignment = list(curr_assignment)
    curr_score = tools['objective']({'assignments': curr_assignment})
    best_score = curr_score
    
    # SA Parameters
    temp = 100.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit_s * 0.95:
        # Neighborhood search: Randomly choose between reassign and swap
        if random.random() < 0.7:
            task = random.randint(0, n - 1)
            new_agent = random.randint(1, m)
            candidate = tools['apply_reassign'](curr_assignment, task, new_agent)
        else:
            t1, t2 = random.sample(range(n), 2)
            candidate = tools['apply_swap_assignments'](curr_assignment, t1, t2)
            
        if candidate:
            cand_score = tools['objective']({'assignments': candidate})
            
            # Acceptance probability: 
            # Note: objective() returns lower is better. 
            # If cand_score < curr_score, delta is negative (always accept).
            delta = cand_score - curr_score
            if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
                curr_assignment = candidate
                curr_score = cand_score
                
                # Global best update
                if curr_score < best_score:
                    best_score = curr_score
                    best_assignment = list(curr_assignment)
        
        # Cool down
        temp *= cooling_rate
        
        # Periodic restart if temperature is too low to escape stagnation
        if temp < 0.01:
            temp = 50.0
            curr_assignment = get_randomized_greedy() or tools['greedy_min_cost']()
            curr_score = tools['objective']({'assignments': curr_assignment})
            
    return {'assignments': best_assignment}