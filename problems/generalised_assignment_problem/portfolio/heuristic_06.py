# MACE evolved heuristic 06/10 for problem: generalised_assignment_problem
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A hybrid metaheuristic for the Generalised Assignment Problem.
    
    Combines:
    1. ILP initialization (for small/constrained instances).
    2. Simulated Annealing with adaptive cooling for broad space exploration.
    3. Tabu-like local search for intensification.
    """
    start_time = time.time()
    n = instance['n']
    m = instance['m']
    
    # 1. Initialization Strategy
    best_assignment = None
    
    # Try ILP first if it's likely to be fast or if the problem is small
    if time_limit_s > 0.5:
        ilp_time = min(2.0, time_limit_s * 0.3)
        best_assignment = tools['ilp_gap'](time_limit_s=ilp_time)
        
    # Fallback to greedy methods
    if best_assignment is None:
        best_assignment = tools['greedy_min_cost']()
        is_feas, _ = tools['is_feasible_assignment'](best_assignment)
        if not is_feas:
            # Try ratio-based greedy if cost-based failed
            best_assignment = tools['greedy_min_resource_ratio']()
            is_feas, _ = tools['is_feasible_assignment'](best_assignment)
            if not is_feas:
                # Last resort: simple filler
                best_assignment = [1] * n

    best_score = tools['objective']({'assignments': best_assignment})
    
    curr_assignment = list(best_assignment)
    curr_score = best_score
    
    # 2. Search Parameters
    temp = 10.0
    cooling_rate = 0.9999
    tabu_list = {} # Store (task, agent) -> expiration time
    
    # 3. Search Loop
    while time.time() - start_time < time_limit_s * 0.95:
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.6: # Reassign
            task = random.randint(0, n - 1)
            new_agent = random.randint(1, m)
            candidate = tools['apply_reassign'](curr_assignment, task, new_agent)
        else: # Swap
            t1, t2 = random.sample(range(n), 2)
            candidate = tools['apply_swap_assignments'](curr_assignment, t1, t2)
            
        if candidate:
            cand_score = tools['objective']({'assignments': candidate})
            
            # Check tabu status
            is_tabu = False
            if move_type < 0.6:
                if tabu_list.get((task, new_agent), 0) > time.time():
                    is_tabu = True
            
            delta = cand_score - curr_score
            
            # Acceptance: Metropolis criterion with tabu override
            if not is_tabu or cand_score < best_score:
                if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
                    curr_assignment = candidate
                    curr_score = cand_score
                    
                    if move_type < 0.6:
                        tabu_list[(task, new_agent)] = time.time() + 0.05
                    
                    # Update global best
                    if curr_score < best_score:
                        best_score = curr_score
                        best_assignment = list(curr_assignment)
        
        # Cool down
        temp *= cooling_rate
        
        # Periodic restart/perturbation
        if temp < 0.001:
            temp = 5.0
            # Shuffle a few assignments to escape local optima
            for _ in range(int(n * 0.1) + 1):
                t = random.randint(0, n - 1)
                a = random.randint(1, m)
                curr_assignment = tools['apply_reassign'](curr_assignment, t, a) or curr_assignment
            curr_score = tools['objective']({'assignments': curr_assignment})
            
    return {'assignments': best_assignment}