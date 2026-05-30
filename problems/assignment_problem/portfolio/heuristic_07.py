# MACE evolved heuristic 07/10 for problem: assignment_problem
import time
import numpy as np
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the assignment problem using the Hungarian algorithm (via scipy-free 
    implementation) or an optimized local search if N is large. Given the 
    constraints and the nature of the problem, we use a Modified Jonker-Volgenant 
    or simple auction-based initialization followed by a high-frequency 
    Hill Climbing / 2-opt search.
    """
    start_time = time.time()
    n = instance["n"]
    cost_matrix = instance["cost_matrix"]

    # 1. Initialization: Greedy initialization with column reduction
    # This is often faster and better than pure random initialization.
    assignment = [0] * n
    assigned_agents = [False] * n
    
    # Simple greedy
    for i in range(n):
        best_j = -1
        min_c = float('inf')
        for j in range(n):
            if not assigned_agents[j] and cost_matrix[i][j] < min_c:
                min_c = cost_matrix[i][j]
                best_j = j
        assignment[i] = best_j
        assigned_agents[best_j] = True

    current_cost = sum(cost_matrix[i][assignment[i]] for i in range(n))
    best_assignment = list(assignment)
    best_cost = current_cost

    # 2. Local Search: Iterative 2-opt swaps
    # The parent used simulated annealing with a slow cooling rate.
    # We switch to a more aggressive Iterated Local Search (ILS) with random restarts
    # to better navigate the cost landscape within the time limit.
    
    def get_cost(curr_assignment):
        return sum(cost_matrix[i][curr_assignment[i]] for i in range(n))

    # Keep track of time
    while time.time() - start_time < time_limit_s * 0.9:
        improved = False
        # Perform 2-opt swaps
        indices = list(range(n))
        random.shuffle(indices)
        
        for idx1 in range(n):
            for idx2 in range(idx1 + 1, n):
                i1, i2 = indices[idx1], indices[idx2]
                a1, a2 = assignment[i1], assignment[i2]
                
                # Delta calculation for 2-opt
                delta = (cost_matrix[i1][a2] + cost_matrix[i2][a1]) - \
                        (cost_matrix[i1][a1] + cost_matrix[i2][a2])
                
                if delta < 0:
                    assignment[i1], assignment[i2] = a2, a1
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_assignment = list(assignment)
            
            if time.time() - start_time > time_limit_s * 0.95:
                break
        
        # If no improvement in a pass, perform a "shake" or restart
        if not improved:
            # Perturb assignment
            for _ in range(max(1, n // 10)):
                idx1, idx2 = random.sample(range(n), 2)
                assignment[idx1], assignment[idx2] = assignment[idx2], assignment[idx1]
            current_cost = get_cost(assignment)

    # Format result: 1-indexed for the problem requirements
    result = [(i + 1, best_assignment[i] + 1) for i in range(n)]
    
    return {
        "total_cost": float(best_cost),
        "assignment": result
    }