# MACE evolved heuristic 08/10 for problem: assortment_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized metaheuristic that combines a robust initial construction 
    with a Simulated Annealing local search over piece count distributions.
    """
    start_time = time.time()
    m = tools['n_types']()
    n_stocks = tools['n_stocks']()
    
    # 1. Start with the best possible baseline
    best_sol = tools['solve_default'](time_limit_s=time_limit_s * 0.2)
    best_waste = best_sol.get('objective', float('inf'))
    
    # Helper to calculate waste for a count distribution
    def evaluate_distribution(counts, stock_type):
        instances, leftover = tools['pack_counts_into_stock'](stock_type, counts)
        if any(leftover.values()):
            return None, None
        
        placements = {i + 1: inst for i, inst in enumerate(instances)}
        _, _, waste = tools['total_waste'](placements)
        return placements, waste

    # 2. Local Search over piece counts and stock types
    # Initial state: counts at the midpoint of min/max
    curr_counts = {t: (tools['piece_type_min'](t) + tools['piece_type_max'](t)) // 2 
                   for t in range(1, m + 1)}
    
    temp = 1.0
    cooling = 0.995
    
    # Time-constrained search loop
    while time.time() - start_time < time_limit_s * 0.9:
        # Move: change one count
        t = random.randint(1, m)
        old_val = curr_counts[t]
        delta = random.choice([-1, 1])
        new_val = max(tools['piece_type_min'](t), min(tools['piece_type_max'](t), old_val + delta))
        curr_counts[t] = new_val
        
        # Select a random stock type
        st = random.randint(1, n_stocks)
        
        placements, waste = evaluate_distribution(curr_counts, st)
        
        if placements is not None:
            # Acceptance logic
            if waste < best_waste:
                best_waste = waste
                best_sol = {'objective': waste, 'placements': placements}
                # Keep the move
            else:
                # Simulated Annealing probability
                prob = math.exp((best_waste - waste) / (temp + 1e-6))
                if random.random() < prob:
                    # Keep the move
                    pass
                else:
                    curr_counts[t] = old_val
        else:
            # Revert if infeasible
            curr_counts[t] = old_val
            
        temp *= cooling
        
    # 3. Final sanity check and fallback
    feasible, _ = tools['is_feasible'](best_sol)
    if not feasible:
        return tools['solve_default'](time_limit_s=min(time_limit_s * 0.1, 1.0))
        
    return best_sol