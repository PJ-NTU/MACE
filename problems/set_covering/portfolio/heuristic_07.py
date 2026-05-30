# MACE evolved heuristic 07/10 for problem: set_covering
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher heuristic for the Set Covering Problem.
    
    Hypothesis:
    - Smaller, dense instances are well-suited for Simulated Annealing (A) 
      as the transition space is small and the energy landscape can be 
      navigated effectively.
    - Larger, sparser instances are better handled by ILP-based LNS (B),
      as the constraint matrix is often sparse enough for the CBC solver
      to find high-quality gaps efficiently.
    """
    start_time = time.time()
    m = instance['m']
    n = instance['n']
    
    # Calculate density: average number of columns covering a row
    total_cover_slots = sum(len(row) for row in instance['row_cover'])
    density = total_cover_slots / (m * n) if (m * n) > 0 else 0
    
    # Heuristic Dispatch:
    # If the problem is small (m*n < 50000) or high density, use SA.
    # Otherwise, use ILP-based LNS which handles large-scale sparsity better.
    if (m * n < 50000) or (density > 0.1):
        return _solve_sa(instance, tools, time_limit_s)
    else:
        return _solve_lns(instance, tools, time_limit_s)

def _solve_sa(instance, tools, time_limit_s):
    start_time = time.time()
    m, n, costs = instance['m'], instance['n'], instance['costs']
    
    current_selection = set(tools['greedy_cover_by_cost_ratio']())
    best_selection = set(current_selection)
    best_cost = tools['cost_of_selection'](list(best_selection))
    
    temp = 100.0
    cooling_rate = 0.9999
    penalty_weight = max(costs) * 2
    
    while time.time() - start_time < time_limit_s * 0.9:
        col = random.randint(1, n)
        if col in current_selection:
            current_selection.remove(col)
        else:
            current_selection.add(col)
            
        uncovered = tools['uncovered_rows'](current_selection)
        curr_energy = tools['cost_of_selection'](list(current_selection)) + (len(uncovered) * penalty_weight)
        
        # Simple local improvement step
        if tools['is_full_cover'](current_selection):
            refined = tools['remove_redundant'](list(current_selection))
            c_cost = tools['cost_of_selection'](refined)
            if c_cost < best_cost:
                best_cost = c_cost
                best_selection = set(refined)
        
        temp *= cooling_rate
        
    return tools['make_solution'](list(best_selection))

def _solve_lns(instance, tools, time_limit_s):
    start_time = time.time()
    best_solution = tools['greedy_cover_by_cost_ratio']()
    best_solution = tools['remove_redundant'](best_solution)
    best_cost = tools['cost_of_selection'](best_solution)
    
    iterations = 0
    while time.time() - start_time < time_limit_s * 0.85 and iterations < 5:
        current_cols = list(best_solution)
        random.shuffle(current_cols)
        # Remove 30% to give ILP more room to optimize
        remove_n = max(1, int(len(current_cols) * 0.3))
        kept_cols = current_cols[remove_n:]
        
        ilp_sol = tools['ilp_solve_cover'](
            must_include=kept_cols, 
            time_limit_s=(time_limit_s - (time.time() - start_time)) * 0.5
        )
        
        if ilp_sol:
            refined = tools['remove_redundant'](ilp_sol)
            cost = tools['cost_of_selection'](refined)
            if cost < best_cost:
                best_cost = cost
                best_solution = refined
        iterations += 1
        
    return tools['make_solution'](best_solution)