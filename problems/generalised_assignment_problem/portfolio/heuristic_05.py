# MACE evolved heuristic 05/10 for problem: generalised_assignment_problem
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Simulated Annealing (SA) heuristic for GAP with an improved
    neighbor selection strategy that prioritizes agents with higher
    remaining capacity to increase the likelihood of feasible moves.
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
        # Weighted Neighborhood search:
        # Instead of uniform random agent, we bias selection towards agents 
        # that have more remaining capacity, as they are more likely 
        # to accept a new task (reassign).
        if random.random() < 0.7:
            task = random.randint(0, n - 1)
            # Calculate remaining capacities for all agents
            rem_caps = [tools['agent_remaining'](i + 1, curr_assignment) for i in range(m)]
            # Use softmax-like probability or weighted choice to pick an agent
            # Adding a small constant to ensure positive weights
            weights = [max(r, 0.001) for r in rem_caps]
            new_agent = random.choices(range(1, m + 1), weights=weights, k=1)[0]
            candidate = tools['apply_reassign'](curr_assignment, task, new_agent)
        else:
            t1, t2 = random.sample(range(n), 2)
            candidate = tools['apply_swap_assignments'](curr_assignment, t1, t2)
            
        if candidate:
            cand_score = tools['objective']({'assignments': candidate})
            
            # Acceptance probability
            delta = cand_score - curr_score
            # Note: objective() returns lower is better, so delta < 0 is an improvement
            if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
                curr_assignment = candidate
                curr_score = cand_score
                
                # Global best update
                if curr_score < best_score:
                    best_score = curr_score
                    best_assignment = list(curr_assignment)
        
        # Cool down
        temp *= cooling_rate
        
        # Periodic restart
        if temp < 0.01:
            temp = 50.0
            curr_assignment = get_randomized_greedy() or tools['greedy_min_cost']()
            curr_score = tools['objective']({'assignments': curr_assignment})
            
    return {'assignments': best_assignment}