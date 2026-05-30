# MACE evolved heuristic 01/10 for problem: set_covering
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Implements a GRASP-based heuristic for the Set Covering Problem.
    1. Generates initial solutions using a randomized greedy cost-effectiveness approach.
    2. Performs local search (removing redundant columns).
    3. Keeps track of the best solution found within the time limit.
    """
    start_time = time.time()
    
    m = instance["m"]
    n = instance["n"]
    costs = instance["costs"]
    
    best_solution = None
    best_cost = float('inf')

    # GRASP parameters
    alpha = 0.2  # Restricted Candidate List parameter

    while time.time() - start_time < time_limit_s * 0.85:
        # Randomized Greedy Construction
        selected = set()
        uncovered = set(range(1, m + 1))
        
        while uncovered:
            # Calculate cost-effectiveness for all potential columns
            candidates = []
            for col in range(1, n + 1):
                covers = tools['column_covers'](col)
                new_rows = len(covers.intersection(uncovered))
                if new_rows > 0:
                    cost = costs[col - 1]
                    ratio = cost / new_rows
                    candidates.append((ratio, col, covers))
            
            if not candidates:
                break
                
            # Sort by ratio
            candidates.sort(key=lambda x: x[0])
            
            # Restricted Candidate List (RCL)
            limit = max(1, int(len(candidates) * alpha))
            idx = random.randint(0, limit - 1)
            _, col, covers = candidates[idx]
            
            selected.add(col)
            uncovered -= covers
        
        # If construction failed to cover everything, skip
        if uncovered:
            continue
            
        # Local Search: Remove redundant columns
        refined = tools['remove_redundant'](list(selected))
        
        # Evaluate
        current_cost = tools['cost_of_selection'](refined)
        if current_cost < best_cost:
            best_cost = current_cost
            best_solution = list(refined)
            
    # Fallback to default if no valid solution found
    if best_solution is None:
        return tools['solve_default'](time_limit_s=max(0.1, time_limit_s * 0.1))
    
    return {"selected_columns": best_solution}