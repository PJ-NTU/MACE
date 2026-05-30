# MACE evolved heuristic 06/10 for problem: assignment_problem
import time
import numpy as np
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A hybrid assignment solver that dispatches based on matrix density/scale.
    
    Heuristic:
    - If the matrix is dense/large (n > 100), use the Auction Algorithm with 
      adaptive epsilon scaling (Parent A) as it handles large-scale stability well.
    - For smaller, potentially more constrained matrices (n <= 100), use an 
      Auction-based initialization followed by a 2-opt local search (Parent B) 
      to refine the solution in the tighter landscape.
    """
    start_time = time.time()
    n = instance["n"]
    cost_matrix = instance["cost_matrix"].astype(float)
    
    # Dispatch logic:
    # Large n benefits from the pure Auction approach (Parent A) due to complexity.
    # Small n benefits from local search refinement (Parent B) to escape local optima.
    if n > 100:
        return _solve_auction_optimized(n, cost_matrix, start_time, time_limit_s)
    else:
        return _solve_auction_with_local_search(n, cost_matrix, start_time, time_limit_s)

def _solve_auction_optimized(n, cost_matrix, start_time, time_limit_s):
    c_min, c_max = np.min(cost_matrix), np.max(cost_matrix)
    if c_max > c_min:
        costs = (cost_matrix - c_min) / (c_max - c_min)
    else:
        costs = cost_matrix
        
    eps = 1.0 / n
    prices = np.zeros(n, dtype=float)
    assignment = -np.ones(n, dtype=int)
    owner = -np.ones(n, dtype=int)
    unassigned = list(range(n))
    
    while unassigned and (time.time() - start_time < time_limit_s * 0.9):
        item = unassigned.pop(0)
        net_values = -costs[item] - prices
        best_idx = np.argpartition(net_values, -2)[-2:]
        best_agent, second_best_agent = best_idx[1], best_idx[0]
        
        bid = net_values[best_agent] - net_values[second_best_agent] + eps
        prices[best_agent] += bid
        old_owner = owner[best_agent]
        owner[best_agent] = item
        assignment[item] = best_agent
        if old_owner != -1:
            unassigned.append(old_owner)
        if len(unassigned) == 0 and eps > 1e-9:
            eps *= 0.5
            unassigned = [i for i in range(n) if assignment[i] == -1]
            
    # Cleanup
    for i in range(n):
        if assignment[i] == -1:
            mask = np.ones(n, dtype=bool)
            mask[assignment[assignment != -1]] = False
            assignment[i] = np.where(mask)[0][0]
            
    total_cost = float(np.sum(cost_matrix[np.arange(n), assignment]))
    return {"total_cost": total_cost, "assignment": [(i + 1, int(assignment[i]) + 1) for i in range(n)]}

def _solve_auction_with_local_search(n, cost_matrix, start_time, time_limit_s):
    assignment = -np.ones(n, dtype=int)
    prices = np.zeros(n, dtype=float)
    owner = -np.ones(n, dtype=int)
    eps = np.max(cost_matrix) / n if n > 0 else 1.0
    unassigned = list(range(n))
    
    while unassigned and (time.time() - start_time < time_limit_s * 0.7):
        item = unassigned.pop(0)
        net_values = -cost_matrix[item] - prices
        best_agent = np.argmax(net_values)
        vals = np.sort(net_values)
        bid = vals[-1] - vals[-2] + eps if n > 1 else eps
        prices[best_agent] += bid
        old_owner = owner[best_agent]
        owner[best_agent] = item
        assignment[item] = best_agent
        if old_owner != -1: unassigned.append(old_owner)
        eps = max(eps * 0.9, 1e-6)
        
    for i in range(n):
        if assignment[i] == -1:
            assigned = set(assignment[assignment != -1])
            for j in range(n):
                if j not in assigned:
                    assignment[i] = j
                    break
                    
    # Local search
    while time.time() < start_time + time_limit_s * 0.95:
        i, j = random.sample(range(n), 2)
        if cost_matrix[i, assignment[j]] + cost_matrix[j, assignment[i]] < \
           cost_matrix[i, assignment[i]] + cost_matrix[j, assignment[j]]:
            assignment[i], assignment[j] = assignment[j], assignment[i]
            
    total_cost = float(np.sum(cost_matrix[np.arange(n), assignment]))
    return {"total_cost": total_cost, "assignment": [(i + 1, int(assignment[i]) + 1) for i in range(n)]}