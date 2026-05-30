# MACE evolved heuristic 03/10 for problem: assortment_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A non-greedy, stochastic metaheuristic that uses a Simulated Annealing 
    approach on the 'Target Piece Count' vector.
    
    Fixed: The returned dictionary must include the 'objective' key 
    to match the expected schema of the evaluators.
    """
    start_time = time.time()
    m = tools['n_types']()
    n_stocks = tools['n_stocks']()
    
    def get_count_vector():
        return {t: random.randint(tools['piece_type_min'](t), tools['piece_type_max'](t)) 
                for t in range(1, m + 1)}

    def pack_counts(counts, stock_type):
        instances, leftover = tools['pack_counts_into_stock'](stock_type, counts)
        if any(leftover.values()):
            return None
        # Must return the full dictionary structure expected by evaluation
        return {'placements': {i + 1: inst for i, inst in enumerate(instances)}}

    # Initial state
    curr_counts = get_count_vector()
    best_sol = None
    best_waste = float('inf')
    
    # Temperature schedule
    temp = 1.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit_s * 0.95:
        # Generate neighbor: modify one count
        t = random.randint(1, m)
        old_val = curr_counts[t]
        delta = random.choice([-1, 1])
        curr_counts[t] = max(tools['piece_type_min'](t), min(tools['piece_type_max'](t), curr_counts[t] + delta))
        
        # Try packing into a random stock type
        st = random.randint(1, n_stocks)
        sol = pack_counts(curr_counts, st)
        
        if sol:
            _, _, waste = tools['total_waste'](sol['placements'])
            # Acceptance probability
            if waste < best_waste:
                best_waste = waste
                best_sol = {**sol, 'objective': waste}
                acceptance = True
            else:
                prob = math.exp((best_waste - waste) / (temp + 1e-9))
                acceptance = (random.random() < prob)
            
            if not acceptance:
                curr_counts[t] = old_val
        else:
            # Backtrack if packing failed
            curr_counts[t] = old_val
            
        temp *= cooling_rate
        
    # If no solution found, fallback to the safest possible method
    if best_sol is None:
        return tools['solve_default'](time_limit_s=max(0.1, time_limit_s * 0.1))
        
    return best_sol