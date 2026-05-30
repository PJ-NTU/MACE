# MACE evolved heuristic 02/10 for problem: assignment_problem
import time
import numpy as np
import random
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Core Logic: Hungarian-inspired Min-Cost Flow relaxation (Auction Algorithm).
    
    The portfolio members primarily use greedy initialization followed by local 
    search (2-opt). The Auction Algorithm differs by using a dual-based 
    approach: agents bid on items, and prices (dual variables) are updated 
    iteratively to reach an equilibrium. This approach is fundamentally 
    different from local search swaps.
    """
    start_time = time.time()
    n = instance["n"]
    cost_matrix = instance["cost_matrix"]
    
    # Epsilon-scaling for the Auction Algorithm
    # As epsilon approaches 0, the solution converges to the optimal
    eps = 1.0 / (n + 1)
    prices = np.zeros(n, dtype=float)
    assignment = -np.ones(n, dtype=int)
    owner = -np.ones(n, dtype=int)
    
    # Unassigned items
    unassigned = list(range(n))
    
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
        if eps > 1e-6:
            eps *= 0.99
            
    # Final cleanup: ensure feasibility if the auction didn't fully converge
    # (Rare, but necessary for strict ISTH compliance)
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