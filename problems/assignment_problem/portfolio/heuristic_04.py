# MACE evolved heuristic 04/10 for problem: assignment_problem
import time
import numpy as np
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for the Assignment Problem.
    
    Heuristic Hypothesis:
    - Small to Medium instances (n < 60) are highly sensitive to local search 
      plateaus; the SA approach (Parent A) is superior due to its ability 
      to escape local optima via thermal fluctuations.
    - Large instances (n >= 60) are better handled by the Auction Algorithm 
      (Parent B), which performs dual-space updates that scale more gracefully 
      with the number of variables than random-walk local search.
    """
    start_time = time.time()
    n = instance["n"]
    cost_matrix = instance["cost_matrix"]

    if n < 60:
        # Parent A: Simulated Annealing for robustness in smaller spaces
        assignment = list(range(n))
        random.shuffle(assignment)
        
        current_cost = sum(cost_matrix[i][assignment[i]] for i in range(n))
        best_assignment = list(assignment)
        best_cost = current_cost
        
        temp = 1.0
        cooling_rate = 0.9999
        
        while time.time() - start_time < time_limit_s * 0.9:
            i1, i2 = random.sample(range(n), 2)
            a1, a2 = assignment[i1], assignment[i2]
            
            delta = (cost_matrix[i1][a2] + cost_matrix[i2][a1]) - \
                    (cost_matrix[i1][a1] + cost_matrix[i2][a2])
            
            if delta < 0 or (temp > 1e-9 and random.random() < np.exp(-delta / temp)):
                assignment[i1], assignment[i2] = a2, a1
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = list(assignment)
            temp *= cooling_rate
            if temp < 1e-7: temp = 0.1
            
        result = [(i + 1, best_assignment[i] + 1) for i in range(n)]
        return {"total_cost": float(best_cost), "assignment": result}
    
    else:
        # Parent B: Auction Algorithm for efficient dual-space optimization in large matrices
        eps = 1.0 / (n + 1)
        prices = np.zeros(n, dtype=float)
        assignment = -np.ones(n, dtype=int)
        owner = -np.ones(n, dtype=int)
        unassigned = list(range(n))
        
        while unassigned and (time.time() - start_time < time_limit_s * 0.9):
            item = unassigned.pop(0)
            net_values = -cost_matrix[item] - prices
            
            best_agent = np.argmax(net_values)
            best_val = net_values[best_agent]
            
            temp_vals = np.concatenate([net_values[:best_agent], net_values[best_agent+1:]])
            second_best_val = np.max(temp_vals) if n > 1 else best_val - 1.0
            
            bid = best_val - second_best_val + eps
            prices[best_agent] += bid
            
            old_owner = owner[best_agent]
            owner[best_agent] = item
            assignment[item] = best_agent
            
            if old_owner != -1:
                unassigned.append(old_owner)
            if eps > 1e-6:
                eps *= 0.99
                
        # Final cleanup for feasibility
        missing = [i for i in range(n) if assignment[i] == -1]
        taken = {assignment[i] for i in range(n) if assignment[i] != -1}
        free = [j for j in range(n) if j not in taken]
        for i in missing:
            assignment[i] = free.pop()
            
        total_cost = sum(cost_matrix[i, assignment[i]] for i in range(n))
        return {
            "total_cost": float(total_cost),
            "assignment": [(i + 1, int(assignment[i]) + 1) for i in range(n)]
        }