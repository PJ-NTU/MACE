# MACE evolved heuristic 02/10 for problem: assortment_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    start_time = time.time()
    
    # 1. Start with a baseline solution using greedy_for_bounds('max')
    # This guarantees min constraints are met and fills comfortably.
    best_solution = tools['solve_default'](time_limit_s=time_limit_s * 0.3)
    
    def get_objective(sol):
        try:
            return tools['objective'](sol)
        except:
            return float('inf')

    best_obj = get_objective(best_solution)
    
    # 2. Local Search: Hill Climbing with small random swaps
    # We attempt to improve waste by swapping piece types if possible
    # or by trying alternative stock types.
    
    stock_types = list(range(1, tools['n_stocks']() + 1))
    piece_types = list(range(1, tools['n_types']() + 1))
    
    while time.time() - start_time < time_limit_s * 0.9:
        # Try a small mutation: swap pieces of different types
        # This helps in re-packing to fill gaps
        p1 = random.choice(piece_types)
        p2 = random.choice(piece_types)
        
        if p1 == p2:
            continue
            
        candidate = tools['apply_swap_pieces'](best_solution, p1, p2)
        
        if candidate:
            candidate_obj = get_objective(candidate)
            if candidate_obj < best_obj:
                best_solution = candidate
                best_obj = candidate_obj
                continue
        
        # Periodically try a fresh greedy construction with a different stock type
        # to explore the limited space of 2-stock-type combinations
        if random.random() < 0.1:
            st = random.choice(stock_types)
            new_sol = tools['greedy_for_bounds'](stock_type=st, prefer='max')
            if new_sol:
                new_obj = get_objective(new_sol)
                if new_obj < best_obj:
                    best_solution = new_sol
                    best_obj = new_obj
        
        # Safety break if we haven't found a solution (though solve_default should)
        if best_obj == float('inf'):
            return tools['greedy_minimal_feasible']()

    # Final check: ensure the solution is feasible before returning
    feasible, _ = tools['is_feasible'](best_solution)
    if not feasible:
        # Fallback to a known feasible default if local search corrupted it
        return tools['solve_default'](time_limit_s=1.0)
        
    return best_solution