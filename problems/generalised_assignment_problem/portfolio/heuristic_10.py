# MACE evolved heuristic 10/10 for problem: generalised_assignment_problem
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Modified VNDS heuristic for GAP.
    Local modification: Improved the repair step by incorporating a 'least-regret' 
    logic during the destruction-repair phase to better handle tight capacity constraints.
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
                if not costs: continue
                
                regret = (costs[1][0] - costs[0][0]) if len(costs) > 1 else float('inf')
                
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
            
        # Repair using Least Regret logic for removed tasks
        rem_caps = [tools['agent_remaining'](i+1, new_assignment) for i in range(m)]
        unassigned = list(tasks_to_remove)
        
        while unassigned:
            best_task = -1
            max_regret = -1
            best_agent = -1
            for t in unassigned:
                feasible = [(instance['cost_matrix'][i-1][t], i) for i in range(1, m+1) 
                            if instance['consumption_matrix'][i-1][t] <= rem_caps[i-1]]
                if not feasible: return None
                feasible.sort()
                regret = (feasible[1][0] - feasible[0][0]) if len(feasible) > 1 else float('inf')
                if regret > max_regret:
                    max_regret, best_task, best_agent = regret, t, feasible[0][1]
            
            if best_task == -1: return None
            new_assignment[best_task] = best_agent
            rem_caps[best_agent-1] -= instance['consumption_matrix'][best_agent-1][best_task]
            unassigned.remove(best_task)
        return new_assignment

    curr_assignment = list(best_assignment)
    while time.time() - start_time < time_limit_s * 0.95:
        intensity = random.uniform(0.1, 0.4)
        candidate = destroy_and_repair(curr_assignment, intensity)
        
        if candidate:
            cand_score = tools['objective']({'assignments': candidate})
            if cand_score < best_score:
                best_score = cand_score
                best_assignment = list(candidate)
                curr_assignment = list(candidate)
            elif random.random() < 0.05:
                curr_assignment = list(candidate)
        else:
            curr_assignment = list(best_assignment)
            
    return {'assignments': best_assignment}