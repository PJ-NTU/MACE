# MACE evolved heuristic 03/10 for problem: generalised_assignment_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Simulated Annealing (SA) heuristic for the GAP.
    
    The portfolio is dominated by greedy construction + local search (Hill Climbing / Tabu).
    This heuristic differs by:
    1. Using a probabilistic acceptance criterion (Simulated Annealing) to escape local optima
       without requiring explicit tabu lists or restarts.
    2. Using a temperature-based schedule to transition from exploration to exploitation.
    3. Not relying on ILP solvers, focusing on pure stochastic search.
    """
    start_time = time.time()
    n = tools['n_tasks']()
    m = tools['n_agents']()
    
    # Construction: Random feasible seed
    # Unlike others, we don't rely on greedy_min_cost. We build by shuffled random assignment.
    def get_random_feasible():
        assignment = [0] * n
        rem_cap = list(instance['capacities'])
        tasks = list(range(n))
        random.shuffle(tasks)
        for j in tasks:
            feasible_agents = [i for i in range(1, m + 1) if instance['consumption_matrix'][i-1][j] <= rem_cap[i-1]]
            if not feasible_agents: return None
            choice = random.choice(feasible_agents)
            assignment[j] = choice
            rem_cap[choice-1] -= instance['consumption_matrix'][choice-1][j]
        return assignment

    # Initial state
    current_assignment = get_random_feasible()
    if not current_assignment:
        current_assignment = tools['greedy_min_cost']()
        
    best_assignment = list(current_assignment)
    best_score = tools['objective']({'assignments': best_assignment})
    
    # SA Parameters
    temp = 100.0
    cooling_rate = 0.99995
    
    # Main loop
    while time.time() - start_time < time_limit_s * 0.95:
        # Generate neighbor
        if random.random() < 0.8:
            # Reassign
            neighbor = tools['apply_reassign'](current_assignment, random.randint(0, n-1), random.randint(1, m))
        else:
            # Swap
            t1, t2 = random.sample(range(n), 2)
            neighbor = tools['apply_swap_assignments'](current_assignment, t1, t2)
        
        if neighbor:
            new_score = tools['objective']({'assignments': neighbor})
            delta = new_score - best_score
            
            # Acceptance probability
            # If delta < 0 (better), accept. If delta > 0, accept with probability exp(-delta/temp)
            if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
                current_assignment = neighbor
                if new_score < best_score:
                    best_score = new_score
                    best_assignment = list(neighbor)
        
        # Cool down
        temp *= cooling_rate
        if temp < 0.001:
            temp = 0.001
            
    return {'assignments': best_assignment}