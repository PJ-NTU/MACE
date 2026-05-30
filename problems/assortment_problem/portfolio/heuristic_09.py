# MACE evolved heuristic 09/10 for problem: assortment_problem
import time
import random
import itertools

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust hybrid heuristic for the Assortment Problem.
    1. Uses ilp_assortment for systematic optimization over promising stock pairs.
    2. Uses greedy_for_bounds as a reliable fallback.
    3. Employs a local search improvement loop via piece swaps.
    """
    start_time = time.time()
    
    # 1. Initial Baseline: solve_default is robust and handles constraints
    best_sol = tools['solve_default'](time_limit_s=time_limit_s * 0.2)
    best_obj = tools['objective'](best_sol)
    
    # 2. Systematic Exploration: ILP on stock type pairs
    # Focus on pairs of stock types to find better area utilization
    n_stocks = tools['n_stocks']()
    stock_types = list(range(1, n_stocks + 1))
    
    # Prioritize combinations by area to find better fits faster
    stock_combinations = list(itertools.combinations_with_replacement(stock_types, 2))
    random.shuffle(stock_combinations)
    
    for pair in stock_combinations:
        if time.time() - start_time > time_limit_s * 0.6:
            break
            
        try:
            cand = tools['ilp_assortment'](time_limit_s=0.5, stock_type_choices=pair)
            if cand:
                cand_obj = tools['objective'](cand)
                if cand_obj < best_obj:
                    best_sol = cand
                    best_obj = cand_obj
        except Exception:
            continue

    # 3. Local Search: Iterative improvement via swaps
    # Small perturbations to improve packing density
    piece_types = list(range(1, tools['n_types']() + 1))
    
    while time.time() - start_time < time_limit_s * 0.95:
        # Randomly select two piece types to swap
        p1, p2 = random.sample(piece_types, 2)
        
        # Apply swap
        candidate = tools['apply_swap_pieces'](best_sol, p1, p2)
        
        if candidate:
            cand_obj = tools['objective'](candidate)
            if cand_obj < best_obj:
                best_sol = candidate
                best_obj = cand_obj
        
        # Periodically try greedy_for_bounds with 'max' to shift towards higher utilization
        if random.random() < 0.05:
            st = random.choice(stock_types)
            new_sol = tools['greedy_for_bounds'](stock_type=st, prefer='max')
            if new_sol and tools['is_feasible'](new_sol)[0]:
                new_obj = tools['objective'](new_sol)
                if new_obj < best_obj:
                    best_sol = new_sol
                    best_obj = new_obj
                    
    # Final validation check
    feasible, _ = tools['is_feasible'](best_sol)
    if not feasible:
        # If something went wrong, return a guaranteed feasible solution
        return tools['solve_default'](time_limit_s=1.0)
        
    return best_sol