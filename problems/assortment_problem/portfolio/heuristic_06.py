# MACE evolved heuristic 06/10 for problem: assortment_problem
import time
import random
import itertools

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved solver: Uses a combination of ILP-based assortment optimization
    and a search over different stock type combinations to minimize waste.
    """
    start_time = time.time()
    
    # 1. Start with a baseline from default solver
    best_sol = tools['solve_default'](time_limit_s=min(time_limit_s * 0.2, 1.0))
    if 'objective' not in best_sol:
        _, _, waste = tools['total_waste'](best_sol['placements'])
        best_sol['objective'] = waste

    # 2. Strategic search over potential stock pairs
    # The objective is to minimize waste, so we explore combinations of 1 or 2 stocks.
    # Larger stocks or combinations that better fit the piece total area are prioritized.
    n_stocks = tools['n_stocks']()
    stock_indices = list(range(1, n_stocks + 1))
    
    # Generate all valid stock pair combinations (1 or 2 distinct types)
    possible_pairs = []
    for r in [1, 2]:
        possible_pairs.extend(list(itertools.combinations(stock_indices, r)))
    
    # Sort pairs by total area to prioritize potentially more efficient packing
    def total_stock_area(pair):
        return sum(tools['stock_area'](s) for s in pair)
    
    possible_pairs.sort(key=total_stock_area, reverse=True)
    
    # 3. Iterate through combinations and optimize
    for pair in possible_pairs:
        if time.time() - start_time > time_limit_s * 0.8:
            break
            
        try:
            # ilp_assortment is the most powerful tool for this specific problem type
            candidate = tools['ilp_assortment'](time_limit_s=0.5, stock_type_choices=pair)
            
            if candidate and 'placements' in candidate:
                # Re-verify and ensure objective is set
                _, _, waste = tools['total_waste'](candidate['placements'])
                candidate['objective'] = waste
                
                # Check feasibility just in case
                is_f, _ = tools['is_feasible'](candidate)
                if is_f and candidate['objective'] < best_sol['objective']:
                    best_sol = candidate
        except Exception:
            continue

    # 4. Final refinement: Small local search
    # If time remains, attempt to improve by swapping pieces
    m = tools['n_types']()
    while time.time() - start_time < time_limit_s * 0.95:
        t1 = random.randint(1, m)
        t2 = random.randint(1, m)
        if t1 == t2:
            continue
            
        # Try a swap attempt
        new_sol = tools['apply_swap_pieces'](best_sol, t1, t2)
        if new_sol:
            _, _, waste = tools['total_waste'](new_sol['placements'])
            if waste < best_sol['objective']:
                new_sol['objective'] = waste
                best_sol = new_sol
        else:
            # If no progress, occasionally break to avoid infinite loops
            if random.random() < 0.05:
                break
                
    return best_sol