# MACE evolved heuristic 03/10 for problem: assignment_problem
import time
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Core Logic: Auction Algorithm with improved epsilon-scaling decay.
    Modification: Adjusted the epsilon decay factor to be more aggressive 
    to ensure faster convergence in tighter time budgets while maintaining 
    precision.
    """
    start_time = time.time()
    n = instance["n"]
    cost_matrix = instance["cost_matrix"]
    
    # Epsilon-scaling for the Auction Algorithm
    # Initial epsilon set to a reasonable magnitude based on cost range
    eps = 1.0
    prices = np.zeros(n, dtype=float)
    assignment = -np.ones(n, dtype=int)
    owner = -np.ones(n, dtype=int)
    
    # Unassigned items
    unassigned = list(range(n))
    
    # Use a faster decay to prioritize finding a feasible, good solution early
    decay = 0.95
    
    while unassigned and (time.time() - start_time < time_limit_s * 0.9):
        item = unassigned.pop(0)
        
        # Find best and second best agents for this item based on net value
        # Net value = -cost - price
        net_values = -cost_matrix[item] - prices
        
        # Get top two agents
        best_agent = np.argmax(net_values)
        best_val = net_values[best_agent]
        
        # Second best value
        temp_vals = np.delete(net_values, best_agent)
        second_best_val = np.max(temp_vals) if n > 1 else best_val - 1.0
        
        # Bid increment
        bid = best_val - second_best_val + eps
        
        # Update price of the best agent
        prices[best_agent] += bid
        
        # Reassign
        old_owner = owner[best_agent]
        owner[best_agent] = item
        assignment[item] = best_agent
        
        if old_owner != -1:
            unassigned.append(old_owner)
            
        # Decay epsilon to refine the solution
        if eps > 1e-7:
            eps *= decay
            
    # Final cleanup: ensure feasibility if the auction didn't fully converge
    missing_items = [i for i in range(n) if assignment[i] == -1]
    taken_agents = {assignment[i] for i in range(n) if assignment[i] != -1}
    free_agents = [j for j in range(n) if j not in taken_agents]
    
    for i in missing_items:
        if free_agents:
            assignment[i] = free_agents.pop()
        else:
            # Fallback to simple greedy if something went wrong
            for j in range(n):
                if j not in taken_agents:
                    assignment[i] = j
                    taken_agents.add(j)
                    break
    
    # Calculate cost
    total_cost = 0.0
    for i in range(n):
        total_cost += cost_matrix[i, assignment[i]]
        
    return {
        "total_cost": float(total_cost),
        "assignment": [(i + 1, int(assignment[i]) + 1) for i in range(n)]
    }