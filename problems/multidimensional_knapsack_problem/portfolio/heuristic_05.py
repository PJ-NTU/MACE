# MACE evolved heuristic 05/10 for problem: multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized Multidimensional Knapsack solver using a focused hybrid strategy:
    1. Multi-start greedy construction for diverse initial coverage.
    2. Local search refinement on all candidates.
    3. An aggressive, time-budget-aware ILP fallback.
    
    Modified: Replaced simple randomized greedy with a 'probabilistic greedy' 
    approach that samples items based on their efficiency (p_i / sum_d r[d][i])
    to explore the search space more effectively.
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # 1. Construction: Diverse greedy starting points.
    candidates = []
    try:
        candidates.append(tools['greedy_by_profit_density']())
        candidates.append(tools['greedy_by_efficiency']())
    except Exception:
        pass
    
    # Probabilistic greedy construction: bias towards higher efficiency
    # but allow randomness to explore different feasible knapsack configurations.
    effs = []
    for i in range(n):
        total_res = sum(tools['item_resource'](i, d) for d in range(tools['n_dims']()))
        effs.append((tools['item_profit'](i) / (total_res + 1e-9), i))
    effs.sort(key=lambda x: x[0], reverse=True)
    
    for _ in range(5):
        sel = []
        # Create a shuffled order that respects efficiency tiers
        indices = [item[1] for item in effs]
        # Introduce noise into the order
        for i in range(len(indices)):
            if random.random() < 0.2:
                swap_idx = random.randint(0, len(indices) - 1)
                indices[i], indices[swap_idx] = indices[swap_idx], indices[i]
        
        for item in indices:
            sel.append(item)
            if not tools['is_within_all_capacities'](sel):
                sel.pop()
        candidates.append(sel)

    best_selection = []
    best_profit = -1.0

    # 2. Refinement: Apply local search to each candidate.
    for sel in candidates:
        elapsed = time.time() - start_time
        remaining = time_limit_s - elapsed
        if remaining < 0.1:
            break
            
        refined = tools['apply_local_swap_in_out'](
            sel, 
            time_limit_s=max(0.05, remaining / (len(candidates) + 1))
        )
        
        profit = tools['profit_of_selection'](refined)
        if profit > best_profit:
            best_profit = profit
            best_selection = refined

    # 3. Final ILP attempt: Use ILP to bridge the gap if time allows.
    remaining_total = time_limit_s - (time.time() - start_time)
    if remaining_total > 0.5:
        ilp_res = tools['ilp_solve_mkp'](
            time_limit_s=remaining_total - 0.1
        )
        if ilp_res is not None:
            ilp_profit = tools['profit_of_selection'](ilp_res)
            if ilp_profit > best_profit:
                best_selection = ilp_res

    # 4. Final output conversion
    x = [0] * n
    for idx in best_selection:
        if 0 <= idx < n:
            x[idx] = 1
            
    return {'x': x}