# MACE evolved heuristic 03/10 for problem: set_covering
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve the Set Covering Problem using a GRASP-inspired metaheuristic:
    1. Start with a greedy cost-effective construction.
    2. Refine using a local search (redundancy removal).
    3. Iterate with limited randomization (Semi-Greedy) until the time limit.
    """
    start_time = time.time()
    
    # 1. Start with a baseline solution using the default solver
    best_solution = tools['solve_default'](time_limit_s=min(time_limit_s * 0.2, 1.0))
    best_cost = tools['objective'](best_solution)
    
    # Pre-calculate useful mappings
    m = instance['m']
    n = instance['n']
    
    # Iterative refinement
    while time.time() - start_time < time_limit_s * 0.85:
        # Construct a randomized greedy solution
        # Instead of picking the absolute best ratio, pick from the top k
        current_selection = set()
        uncovered = set(range(1, m + 1))
        
        # Simple randomized greedy: pick from top 3 best ratios
        while uncovered:
            best_candidates = []
            # Optimization: only look at columns covering the first uncovered row
            target_row = next(iter(uncovered))
            potential_cols = tools['columns_covering_row'](target_row)
            
            # Calculate cost-effectiveness: cost / (new rows covered)
            ratios = []
            for col in potential_cols:
                covers = tools['column_covers'](col)
                new_covered = len(covers.intersection(uncovered))
                if new_covered > 0:
                    ratios.append((tools['column_cost'](col) / new_covered, col))
            
            if not ratios:
                break
                
            ratios.sort()
            # Pick from top 3 (or fewer)
            pick = random.choice(ratios[:3])[1]
            current_selection.add(pick)
            uncovered -= tools['column_covers'](pick)
            
        # Clean up the solution
        refined_list = tools['remove_redundant'](list(current_selection))
        new_solution = tools['make_solution'](refined_list)
        
        # Check feasibility and quality
        is_feas, _ = tools['is_feasible'](new_solution)
        if is_feas:
            cost = tools['objective'](new_solution)
            if cost < best_cost:
                best_cost = cost
                best_solution = new_solution
        
        # Safety break
        if time.time() - start_time > time_limit_s * 0.95:
            break
            
    return best_solution