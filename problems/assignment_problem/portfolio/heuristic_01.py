# MACE evolved heuristic 01/10 for problem: assignment_problem
import time
import numpy as np
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized Assignment Problem solver using a Regret-based construction
    followed by a high-performance Simulated Annealing loop.
    """
    start_time = time.time()
    n = instance["n"]
    cost_matrix = instance["cost_matrix"]

    # 1. Greedy Regret-based Construction
    # This provides a strong starting point by prioritizing assignments that
    # would be most costly if missed.
    assignment = [0] * n  # assignment[i] = agent_index (0-indexed)
    assigned_agents = [False] * n
    
    # Pre-calculate costs for faster access
    # We use a greedy approach with regret heuristics
    items_to_assign = list(range(n))
    while items_to_assign:
        best_regret = -1.0
        best_item = -1
        best_agent = -1
        
        for i in items_to_assign:
            costs = []
            for j in range(n):
                if not assigned_agents[j]:
                    costs.append((cost_matrix[i][j], j))
            
            costs.sort(key=lambda x: x[0])
            
            if len(costs) == 1:
                regret = 0.0
            else:
                regret = costs[1][0] - costs[0][0]
            
            if regret >= best_regret:
                best_regret = regret
                best_item = i
                best_agent = costs[0][1]
        
        assignment[best_item] = best_agent
        assigned_agents[best_agent] = True
        items_to_assign.remove(best_item)

    current_cost = sum(cost_matrix[i][assignment[i]] for i in range(n))
    best_assignment = list(assignment)
    best_cost = current_cost

    # 2. Simulated Annealing
    # Optimized loop with fast delta computation and adaptive temperature
    # The temperature control is tuned to explore early and refine late.
    temp = 1.0
    cooling_rate = 0.9999
    
    # Pre-calculate indices to sample
    idx_range = list(range(n))
    
    while time.time() - start_time < time_limit_s * 0.95:
        # 2-opt swap of two random assignments
        i1, i2 = random.sample(idx_range, 2)
        a1, a2 = assignment[i1], assignment[i2]
        
        # Calculate change in cost
        delta = (cost_matrix[i1][a2] + cost_matrix[i2][a1]) - \
                (cost_matrix[i1][a1] + cost_matrix[i2][a2])
        
        # Metropolis acceptance criterion
        if delta < 0 or (temp > 1e-9 and random.random() < np.exp(-delta / temp)):
            assignment[i1], assignment[i2] = a2, a1
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_assignment = list(assignment)
        
        # Annealing schedule
        temp *= cooling_rate
        
        # Periodically reset temperature if we stagnate
        if temp < 1e-7:
            temp = 0.1

    # Format result: 1-indexed for the problem requirements
    result = [(i + 1, best_assignment[i] + 1) for i in range(n)]
    
    return {
        "total_cost": float(best_cost),
        "assignment": result
    }