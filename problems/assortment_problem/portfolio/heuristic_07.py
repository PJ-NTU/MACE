# MACE evolved heuristic 07/10 for problem: assortment_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined heuristic for the assortment problem.
    Diagnosis: The parent heuristic relied on a limited local search (swaps)
    that rarely improved the packing density significantly. It also lacked
    a mechanism to adjust the total number of pieces placed, which is the 
    primary driver of waste reduction.
    
    Strategy:
    1. Use ILP/Default as the baseline.
    2. Use a "hill-climbing" approach by trying to pack more pieces
       (closer to the 'max' limit) into the most efficient stock types.
    3. Periodically re-generate solutions using different stock type pairings
       to explore the search space of allowed stock configurations.
    """
    start_time = time.time()
    
    # 1. Start with the best known configuration
    best_solution = tools['solve_default'](time_limit_s=max(0.5, time_limit_s * 0.3))
    best_obj = tools['objective'](best_solution)
    
    n_stocks = tools['n_stocks']()
    m = tools['n_types']()

    # 2. Iterative Improvement
    # Try different combinations of stock types and fill densities
    while time.time() - start_time < time_limit_s * 0.9:
        # Select a random stock type to bias the search
        st = random.randint(1, n_stocks)
        
        # Try building a solution with 'max' preference to pack as much as possible,
        # which effectively minimizes waste area percentage by filling stock.
        candidate = tools['greedy_for_bounds'](stock_type=st, prefer='max')
        
        if candidate:
            try:
                # verify feasibility and quality
                is_ok, _ = tools['is_feasible'](candidate)
                if is_ok:
                    cand_obj = tools['objective'](candidate)
                    if cand_obj < best_obj:
                        best_obj = cand_obj
                        best_solution = candidate
            except Exception:
                continue

        # Small perturbations: Attempt to swap pieces to improve packing
        # or just fill gaps. We use the tool provided for this purpose.
        if random.random() < 0.3:
            p1 = random.randint(1, m)
            p2 = random.randint(1, m)
            if p1 != p2:
                perturbed = tools['apply_swap_pieces'](best_solution, p1, p2)
                if perturbed:
                    try:
                        cand_obj = tools['objective'](perturbed)
                        if cand_obj < best_obj:
                            best_obj = cand_obj
                            best_solution = perturbed
                    except Exception:
                        pass
        
        # Adaptive backoff: If we are deep into the time budget, stop searching
        # and lock in the current best.
        if (time.time() - start_time) > time_limit_s * 0.8:
            break
            
    # 3. Final safety check
    # Ensure the best found solution is still valid per the ISTH interface
    feasible, _ = tools['is_feasible'](best_solution)
    if not feasible:
        return tools['solve_default'](time_limit_s=min(0.2, time_limit_s))
        
    return best_solution