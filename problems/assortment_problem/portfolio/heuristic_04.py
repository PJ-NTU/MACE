# MACE evolved heuristic 04/10 for problem: assortment_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Partition-Based Adaptive Search' heuristic.
    
    Unlike the portfolio which relies heavily on local swap-based hill climbing or 
    ILP, this approach uses a 'Bin-Packing Partitioning' strategy.
    
    Core logic:
    1. Instead of swapping individual pieces, we solve the 'Assortment' problem 
       by partitioning the required piece-counts into two distinct 'bins' 
       corresponding to the two available stock types.
    2. It uses a Randomized Greedy constructive approach, then applies a 
       'Metropolis-Hastings' style perturbation to the global piece-count 
       distribution rather than local geometric swaps.
    """
    start_time = time.time()
    
    m = tools['n_types']()
    n_stocks = tools['n_stocks']()
    
    def get_random_counts():
        return {t: random.randint(tools['piece_type_min'](t), tools['piece_type_max'](t)) 
                for t in range(1, m + 1)}

    def build_solution(counts):
        # Pick two distinct stock types
        stocks = random.sample(range(1, n_stocks + 1), min(2, n_stocks))
        # Distribute counts among these stocks
        # Simplified: put all in the first, try to pack, then second.
        # This is very different from the swap-based portfolio approaches.
        res_placements = {}
        leftover = counts
        current_id = 1
        
        for st in stocks:
            instances, leftover = tools['pack_counts_into_stock'](st, leftover)
            for inst in instances:
                res_placements[current_id] = inst
                current_id += 1
            if not any(leftover.values()):
                break
        
        if any(leftover.values()):
            return None
            
        return {'placements': res_placements}

    best_sol = None
    best_waste = float('inf')
    
    # Adaptive Search: Move through the space of valid piece-count vectors
    # using a 'Large Neighborhood Search' (LNS) style perturbation.
    curr_counts = get_random_counts()
    
    while time.time() - start_time < time_limit_s * 0.9:
        # Perturb the entire count vector by a random percentage
        if random.random() < 0.3:
            curr_counts = get_random_counts()
        else:
            t = random.randint(1, m)
            curr_counts[t] = random.randint(tools['piece_type_min'](t), tools['piece_type_max'](t))
            
        sol = build_solution(curr_counts)
        
        if sol:
            _, _, waste = tools['total_waste'](sol['placements'])
            # We don't use swap-based hill climbing, we use a global greedy 
            # construction that is re-evaluated every iteration.
            if waste < best_waste:
                best_waste = waste
                best_sol = {**sol, 'objective': waste}
                
    if best_sol is None:
        return tools['solve_default'](time_limit_s=max(0.1, time_limit_s * 0.1))
        
    return best_sol