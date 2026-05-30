# MACE evolved heuristic 09/10 for problem: generalised_assignment_problem
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Simulated Annealing (SA) heuristic for GAP.
    
    Modified the neighborhood selection to favor 'swap' moves when the 
    current solution is already high-quality, as swaps are more likely to 
    maintain feasibility while exploring tighter regions of the search space.
    """
    start_time = time.time()
    n = instance['n']
    m = instance['m']
    
    # Construction: Randomized Greedy to provide a diverse starting point
    def get_randomized_greedy():
        assignment = [0] * n
        rem_cap = list(instance['capacities'])
        tasks = list(range(n))
        random.shuffle(tasks)
        for j in tasks:
            feasible = [i for i in range(1, m + 1) if instance['consumption_matrix'][i-1][j] <= rem_cap[i-1]]
            if not feasible: return None
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
        # Neighborhood search: Adaptive selection. 
        # Increase swap probability as temperature drops to refine tight constraints.
        swap_prob = 0.3 + (1.0 - min(1.0, temp / 100.0)) * 0.4
        
        if random.random() > swap_prob:
            task = random.randint(0, n - 1)
            new_agent = random.randint(1, m)
            candidate = tools['apply_reassign'](curr_assignment, task, new_agent)
        else:
            t1, t2 = random.sample(range(n), 2)
            candidate = tools['apply_swap_assignments'](curr_assignment, t1, t2)
            
        if candidate:
            cand_score = tools['objective']({'assignments': candidate})
            
            delta = cand_score - curr_score
            if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
                curr_assignment = candidate
                curr_score = cand_score
                
                if curr_score < best_score:
                    best_score = curr_score
                    best_assignment = list(curr_assignment)
        
        temp *= cooling_rate
        
        if temp < 0.01:
            temp = 50.0
            curr_assignment = get_randomized_greedy() or tools['greedy_min_cost']()
            curr_score = tools['objective']({'assignments': curr_assignment})
            
    return {'assignments': best_assignment}