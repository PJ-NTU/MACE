# MACE evolved heuristic 01/10 for problem: assortment_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Heuristic: Greedy construction followed by iterative improvement.
    Uses 'greedy_for_bounds' as base, then attempts to improve waste percentage
    by exploring different stock combinations or packing strategies.
    """
    start_time = time.time()
    
    # 1. Start with a strong baseline
    best_sol = tools['solve_default'](time_limit_s=time_limit_s * 0.5)
    
    # 2. Local Search / Improvement Loop
    # We try to find a better configuration by iterating through different 
    # stock type combinations if time permits.
    n_stocks = tools['n_stocks']()
    stock_combinations = []
    
    # Generate all pairs of stocks
    for i in range(1, n_stocks + 1):
        for j in range(i, n_stocks + 1):
            stock_combinations.append((i, j))
    
    random.shuffle(stock_combinations)
    
    for stock_pair in stock_combinations:
        if time.time() - start_time > time_limit_s * 0.95:
            break
            
        # Attempt to create a solution using these two stock types
        # We use a simple greedy approach: pack all 'max' pieces if possible,
        # otherwise pack 'min' pieces.
        try:
            # We attempt to use the ilp_assortment tool if available
            cand = tools['ilp_assortment'](time_limit_s=0.5, stock_type_choices=stock_pair)
            
            if cand is not None:
                # Validate feasibility
                feasible, _ = tools['is_feasible'](cand)
                if feasible:
                    # Check objective
                    if cand.get('objective', float('inf')) < best_sol.get('objective', float('inf')):
                        best_sol = cand
        except Exception:
            continue
            
    # 3. Final safety check
    # Ensure the returned solution is definitely feasible
    feasible, msg = tools['is_feasible'](best_sol)
    if not feasible:
        # Fallback to the most reliable method if the improved one failed
        return tools['greedy_for_bounds'](prefer='min')
        
    return best_sol