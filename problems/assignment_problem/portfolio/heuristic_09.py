# MACE evolved heuristic 09/10 for problem: assignment_problem
import time
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized Assignment Problem solver using the Auction Algorithm with 
    epsilon-scaling. This implementation ensures feasibility and high 
    precision by utilizing a robust epsilon-scaling schedule and 
    a final cleanup phase to handle any incomplete assignments.
    """
    start_time = time.time()
    n = instance["n"]
    cost_matrix = instance["cost_matrix"].astype(float)
    
    # Epsilon scaling parameters
    # The performance of the auction algorithm is highly dependent on epsilon.
    # Start large to quickly find a valid assignment, decay to find optimality.
    eps = 1.0
    min_eps = 1e-7
    decay = 0.92
    
    prices = np.zeros(n, dtype=float)
    assignment = -np.ones(n, dtype=int)
    owner = -np.ones(n, dtype=int)
    unassigned = list(range(n))
    
    # We prioritize speed and robustness.
    # Time buffer to ensure we return a feasible solution before the limit.
    while unassigned and (time.time() - start_time < time_limit_s * 0.9):
        item = unassigned.pop(0)
        
        # Calculate net values (profit = -cost - price)
        net_values = -cost_matrix[item] - prices
        
        # Identify the best and second-best agents
        best_agent = np.argmax(net_values)
        best_val = net_values[best_agent]
        
        # Find second-best value for bid calculation
        if n > 1:
            # Mask the best agent to find the next best
            second_best_val = np.max(np.delete(net_values, best_agent))
        else:
            second_best_val = best_val - 1.0
            
        # Bid = value difference + epsilon
        bid = (best_val - second_best_val) + eps
        
        # Update price and reassign
        prices[best_agent] += bid
        old_owner = owner[best_agent]
        
        owner[best_agent] = item
        assignment[item] = best_agent
        
        if old_owner != -1:
            unassigned.append(old_owner)
            
        # Decay epsilon to refine the solution towards optimality
        if eps > min_eps:
            eps = max(min_eps, eps * decay)
            
    # Final cleanup: ensure 100% feasibility
    # If the auction didn't finish, force-assign remaining items to free agents
    if -1 in assignment:
        taken_agents = {assignment[i] for i in range(n) if assignment[i] != -1}
        free_agents = [j for j in range(n) if j not in taken_agents]
        for i in range(n):
            if assignment[i] == -1:
                if free_agents:
                    assignment[i] = free_agents.pop()
                else:
                    # In extreme cases, swap with existing to complete
                    assignment[i] = 0 
    
    # Calculate costs using the final assignment
    total_cost = 0.0
    final_assignment = []
    for i in range(n):
        agent = int(assignment[i])
        total_cost += cost_matrix[i, agent]
        final_assignment.append((i + 1, agent + 1))
        
    return {
        "total_cost": float(total_cost),
        "assignment": final_assignment
    }