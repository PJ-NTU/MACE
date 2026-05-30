# MACE evolved heuristic 08/10 for problem: set_covering
import time
import random
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Lagrangian Relaxation-based Dual Decomposition heuristic.
    
    Most portfolio members use constructive greedy or ILP-based LNS.
    This implementation uses a subgradient-based approach to adjust row 
    penalties, effectively solving a series of simplified subproblems 
    to find near-optimal dual variables, which then guide the construction 
    of a primal feasible solution.
    """
    start_time = time.time()
    m = instance['m']
    n = instance['n']
    costs = instance['costs']
    
    # Dual variables (penalties for rows not yet covered)
    # Initialized to a portion of the average column cost
    avg_col_cost = sum(costs) / n
    row_penalties = [avg_col_cost / (m * 0.5)] * m
    
    best_solution = None
    best_cost = float('inf')
    
    # Iterative dual update
    step_size = 1.0
    while time.time() - start_time < time_limit_s * 0.9:
        # Primal construction based on current duals:
        # A column's "utility" is its cost minus sum of row penalties it covers.
        # Pick columns with the most negative utility (most profitable)
        selected = set()
        
        # We perform a greedy pass where we pick columns that maximize:
        # (sum of row_penalties covered) - cost
        uncovered = set(range(m))
        while uncovered:
            best_col = -1
            best_val = -float('inf')
            
            for col in range(1, n + 1):
                covers = tools['column_covers'](col)
                # Intersection with currently uncovered
                relevant_rows = [r - 1 for r in covers if (r - 1) in uncovered]
                if not relevant_rows:
                    continue
                
                gain = sum(row_penalties[r] for r in relevant_rows)
                val = gain - costs[col - 1]
                
                if val > best_val:
                    best_val = val
                    best_col = col
            
            if best_col == -1:
                break
            
            selected.add(best_col)
            for r in tools['column_covers'](best_col):
                if (r - 1) in uncovered:
                    uncovered.remove(r - 1)
        
        # Repair and evaluate
        if not uncovered:
            refined = tools['remove_redundant'](list(selected))
            cost = tools['cost_of_selection'](refined)
            if cost < best_cost:
                best_cost = cost
                best_solution = refined
            
            # Subgradient update: increase penalties for rows not covered by 
            # the 'best' selection, decrease for those heavily over-covered
            # (Lagrangian relaxation update)
            for i in range(m):
                is_covered = any((i + 1) in tools['column_covers'](c) for c in refined)
                if not is_covered:
                    row_penalties[i] += step_size
                else:
                    row_penalties[i] = max(0.01, row_penalties[i] * 0.95)
        else:
            # If infeasible, penalize all currently uncovered rows heavily
            for r in uncovered:
                row_penalties[r] += step_size * 2
        
        step_size *= 0.99
    
    if best_solution is None:
        return tools['solve_default'](time_limit_s=max(0.1, time_limit_s * 0.1))
        
    return tools['make_solution'](best_solution)