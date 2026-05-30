# MACE evolved heuristic 07/10 for problem: generalised_assignment_problem
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Variable Neighborhood Decomposition Search (VNDS) heuristic for GAP.
    
    Unlike the portfolio which relies on simple Hill Climbing, Tabu, or SA 
    with standard reassign/swap moves, this heuristic:
    1. Uses a 'destroy and repair' mechanism (Large Neighborhood Search) 
       where a random subset of tasks is unassigned and then greedily 
       re-inserted according to least-regret criteria.
    2. Does not rely on ILP solvers, avoiding memory/time overhead.
    3. Uses a systematic neighborhood change strategy (VND) to alternate 
       between small perturbations and large-scale reconstruction.
    """
    start_time = time.time()
    n = tools['n_tasks']()
    m = tools['n_agents']()
    
    # Construction: Greedy using Least Regret
    def get_least_regret_assignment():
        assignments = [0] * n
        rem_caps = list(instance['capacities'])
        unassigned = list(range(n))
        
        while unassigned:
            best_task = -1
            max_regret = -1
            best_agent = -1
            
            for j in unassigned:
                costs = sorted([(instance['cost_matrix'][i-1][j], i) for i in range(1, m+1) 
                                if instance['consumption_matrix'][i-1][j] <= rem_caps[i-1]])
                if not costs: continue # Infeasible in this branch
                
                if len(costs) == 1:
                    regret = float('inf')
                else:
                    regret = costs[1][0] - costs[0][0]
                
                if regret > max_regret:
                    max_regret = regret
                    best_task = j
                    best_agent = costs[0][1]
            
            if best_task == -1: break
            assignments[best_task] = best_agent
            rem_caps[best_agent-1] -= instance['consumption_matrix'][best_agent-1][best_task]
            unassigned.remove(best_task)
            
        return assignments if all(a > 0 for a in assignments) else tools['greedy_min_cost']()

    best_assignment = get_least_regret_assignment()
    best_score = tools['objective']({'assignments': best_assignment})
    
    # Neighborhoods
    def destroy_and_repair(curr_assignment, percentage=0.2):
        new_assignment = list(curr_assignment)
        tasks_to_remove = random.sample(range(n), max(1, int(n * percentage)))
        for t in tasks_to_remove:
            new_assignment[t] = 0
            
        # Repair greedily
        rem_caps = [tools['agent_remaining'](i+1, new_assignment) for i in range(m)]
        for t in tasks_to_remove:
            feasible = [i for i in range(1, m+1) if instance['consumption_matrix'][i-1][t] <= rem_caps[i-1]]
            if not feasible: return None
            choice = min(feasible, key=lambda i: instance['cost_matrix'][i-1][t])
            new_assignment[t] = choice
            rem_caps[choice-1] -= instance['consumption_matrix'][choice-1][t]
        return new_assignment

    # Main loop: Alternating LNS phases
    curr_assignment = list(best_assignment)
    while time.time() - start_time < time_limit_s * 0.95:
        # Vary intensity of destruction
        intensity = random.uniform(0.1, 0.4)
        candidate = destroy_and_repair(curr_assignment, intensity)
        
        if candidate:
            cand_score = tools['objective']({'assignments': candidate})
            # Accept if better
            if cand_score < best_score:
                best_score = cand_score
                best_assignment = list(candidate)
                curr_assignment = list(candidate)
            # Occasional non-improving move to escape local minima
            elif random.random() < 0.05:
                curr_assignment = list(candidate)
        else:
            # If repair failed, revert to best
            curr_assignment = list(best_assignment)
            
    return {'assignments': best_assignment}