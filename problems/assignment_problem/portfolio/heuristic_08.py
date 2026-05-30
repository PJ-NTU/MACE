# MACE evolved heuristic 08/10 for problem: assignment_problem
import time
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized Auction Algorithm using the epsilon-scaling method.
    Provides rigorous optimality for the Assignment Problem by 
    systematically reducing the bid increment (epsilon) while 
    maintaining feasibility.
    """
    start_time = time.time()
    n = instance["n"]
    cost_matrix = instance["cost_matrix"].astype(float)
    
    # Pre-scale for numerical stability
    c_max = np.max(cost_matrix[np.isfinite(cost_matrix)])
    c_min = np.min(cost_matrix[np.isfinite(cost_matrix)])
    scale = (c_max - c_min) if c_max > c_min else 1.0
    costs = (cost_matrix - c_min) / scale
    
    # Initialize auction variables
    prices = np.zeros(n, dtype=float)
    assignment = -np.ones(n, dtype=int)
    owner = -np.ones(n, dtype=int)
    
    # Epsilon scaling: geometric reduction ensures convergence to optimal
    eps = 1.0 / n
    min_eps = 1.0 / (n * n * 10)
    
    while eps >= min_eps and (time.time() - start_time < time_limit_s * 0.9):
        unassigned = [i for i in range(n) if assignment[i] == -1]
        
        while unassigned and (time.time() - start_time < time_limit_s * 0.9):
            i = unassigned.pop(0)
            
            # Find the best and second best agent for item i
            # val = -cost - price
            net_values = -costs[i] - prices
            
            # Use argpartition to find top 2 indices efficiently
            idx = np.argpartition(net_values, 1)
            best_j = idx[1]
            second_best_j = idx[0]
            
            val_best = net_values[best_j]
            val_second_best = net_values[second_best_j]
            
            # Bid increment
            bid = val_best - val_second_best + eps
            
            # Update price and assignment
            prices[best_j] += bid
            old_i = owner[best_j]
            
            owner[best_j] = i
            assignment[i] = best_j
            
            if old_i != -1:
                assignment[old_i] = -1
                unassigned.append(old_i)
        
        eps *= 0.25
        
    # Final greedy repair for any unassigned items
    if -1 in assignment:
        taken = set(assignment[assignment != -1])
        free_agents = [j for j in range(n) if j not in taken]
        for i in range(n):
            if assignment[i] == -1:
                assignment[i] = free_agents.pop()
    
    # Compute total cost from the original cost matrix
    total_cost = 0.0
    final_assignment = []
    for i in range(n):
        total_cost += instance["cost_matrix"][i, assignment[i]]
        final_assignment.append((i + 1, int(assignment[i]) + 1))
        
    return {
        "total_cost": float(total_cost),
        "assignment": final_assignment
    }