# MACE evolved heuristic 10/10 for problem: assortment_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Heuristic: Greedy construction followed by iterative improvement.
    Modified: Prioritizes local search by allocating more time to the ILP-based
    exploration and using greedy_for_bounds('max') as a robust fallback.
    """
    start_time = time.time()
    
    # 1. Start with a solid baseline using max packing to fill area efficiently
    best_sol = tools['greedy_for_bounds'](prefer='max')
    
    # 2. Local Search / Improvement Loop
    # Explore stock combinations with more focus on the time budget
    n_stocks = tools['n_stocks']()
    stock_combinations = []
    
    for i in range(1, n_stocks + 1):
        for j in range(i, n_stocks + 1):
            stock_combinations.append((i, j))
    
    random.shuffle(stock_combinations)
    
    # Allocate 85% of time to searching for optimal stock pairings
    search_limit = time_limit_s * 0.85
    
    for stock_pair in stock_combinations:
        if time.time() - start_time > search_limit:
            break
            
        try:
            # ilp_assortment is the most effective way to optimize waste
            # We allow it a larger portion of the remaining time per step
            cand = tools['ilp_assortment'](time_limit_s=max(0.1, (search_limit - (time.time() - start_time)) / 3), 
                                           stock_type_choices=stock_pair)
            
            if cand is not None:
                # Direct check against objective (lower is better)
                if cand.get('objective', float('inf')) < best_sol.get('objective', float('inf')):
                    # Re-verify feasibility just in case
                    feasible, _ = tools['is_feasible'](cand)
                    if feasible:
                        best_sol = cand
        except Exception:
            continue
            
    # 3. Final safety check
    # Ensure the returned solution is definitely feasible
    feasible, msg = tools['is_feasible'](best_sol)
    if not feasible:
        # Fallback to the most reliable minimal feasible construction
        return tools['greedy_minimal_feasible']()
        
    return best_sol