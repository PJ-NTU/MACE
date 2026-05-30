# MACE evolved heuristic 05/10 for problem: assignment_problem
import time
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized Auction Algorithm with adaptive epsilon scaling and 
    pre-processing to handle large cost variations effectively.
    """
    start_time = time.time()
    n = instance["n"]
    cost_matrix = instance["cost_matrix"].astype(float)
    
    # 1. Pre-processing: Scale costs to avoid precision issues with large values
    # and normalize slightly to stabilize the auction.
    c_min, c_max = np.min(cost_matrix), np.max(cost_matrix)
    if c_max > c_min:
        cost_matrix = (cost_matrix - c_min) / (c_max - c_min)
    
    # 2. Initial Epsilon: Start with a wider epsilon based on n
    # This allows for faster initial convergence.
    eps = 1.0 / n
    prices = np.zeros(n, dtype=float)
    assignment = -np.ones(n, dtype=int)
    owner = -np.ones(n, dtype=int)
    
    unassigned = list(range(n))
    
    # 3. Auction Loop
    # We use a balanced strategy: decay epsilon only after a full pass of unassigned
    # to maintain consistency and avoid premature convergence to sub-optimal local minima.
    while unassigned and (time.time() - start_time < time_limit_s * 0.95):
        item = unassigned.pop(0)
        
        # Net value: -cost - price
        net_values = -cost_matrix[item] - prices
        
        # Find best and second best
        best_idx = np.argpartition(net_values, -2)[-2:]
        best_agent = best_idx[1]
        second_best_agent = best_idx[0]
        
        best_val = net_values[best_agent]
        second_best_val = net_values[second_best_agent]
        
        # Bid increment
        bid = best_val - second_best_val + eps
        
        # Update price
        prices[best_agent] += bid
        
        # Reassign
        old_owner = owner[best_agent]
        owner[best_agent] = item
        assignment[item] = best_agent
        
        if old_owner != -1:
            unassigned.append(old_owner)
            
        # Adaptive epsilon decay: slower, more rigorous convergence
        if len(unassigned) == 0 and eps > 1e-9:
            eps *= 0.5
            unassigned = [i for i in range(n) if assignment[i] == -1]
            
    # 4. Feasibility Enforcement
    missing = [i for i in range(n) if assignment[i] == -1]
    if missing:
        taken = set(assignment[assignment != -1])
        free = [j for j in range(n) if j not in taken]
        for i in missing:
            assignment[i] = free.pop()
    
    # Calculate actual cost (using the original cost matrix)
    orig_costs = instance["cost_matrix"]
    total_cost = 0.0
    for i in range(n):
        total_cost += orig_costs[i, assignment[i]]
        
    return {
        "total_cost": float(total_cost),
        "assignment": [(i + 1, int(assignment[i]) + 1) for i in range(n)]
    }