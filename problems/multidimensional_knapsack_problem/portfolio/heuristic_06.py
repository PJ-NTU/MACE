# MACE evolved heuristic 06/10 for problem: multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Redesigned solver for the Multidimensional Knapsack Problem.
    
    Diagnosis of parent:
    1. Over-reliance on local search without global guidance (no ILP usage).
    2. Inefficient LNS: random destruction is often destructive to high-value clusters.
    3. Failure to leverage the provided `ilp_solve_mkp` tool, which is typically 
       the most powerful component for small-to-medium instances or sub-problems.
    
    Strategy:
    1. Use ILP as the primary engine. It provides the best optimality guarantees.
    2. If the problem is too large for a single full ILP solve, use a hybrid 
       approach: start with greedy, then perform LNS using ILP to solve 
       restricted sub-problems (fixing most variables, optimizing a subset).
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # 1. Start with the best available greedy baseline.
    greedy_density = tools['greedy_by_profit_density']()
    greedy_efficiency = tools['greedy_by_efficiency']()
    
    best_selection = greedy_density if tools['profit_of_selection'](greedy_density) > \
                                       tools['profit_of_selection'](greedy_efficiency) else greedy_efficiency
    
    # 2. Attempt to solve the full instance via ILP if time allows.
    # ILP is generally superior to local search for MKP.
    ilp_res = tools['ilp_solve_mkp'](time_limit_s=min(time_limit_s * 0.7, 5.0))
    if ilp_res is not None:
        best_selection = ilp_res
    
    # 3. If time remains, perform a focused LNS refinement using the ILP solver
    # on small windows of variables to improve the current best_selection.
    while time.time() - start_time < time_limit_s * 0.9:
        # Pick a random subset of items to fix (keep most, optimize some)
        # Fix 80% of items to their current state in best_selection
        all_items = set(range(n))
        to_optimize = random.sample(range(n), min(n, 20)) # Small window for ILP
        
        must_include = []
        must_exclude = []
        
        for idx in to_optimize:
            if idx in best_selection:
                # We can choose to keep or flip, by not constraining
                pass
            else:
                pass
        
        # Constrain everything outside the window to current state
        fixed_items = all_items - set(to_optimize)
        for idx in fixed_items:
            if idx in best_selection:
                must_include.append(idx)
            else:
                must_exclude.append(idx)
        
        sub_res = tools['ilp_solve_mkp'](
            must_include=must_include, 
            must_exclude=must_exclude, 
            time_limit_s=min(1.0, (time_limit_s - (time.time() - start_time)) / 2)
        )
        
        if sub_res is not None:
            if tools['profit_of_selection'](sub_res) > tools['profit_of_selection'](best_selection):
                best_selection = sub_res
        else:
            break
            
    # Final cleanup: ensure local optimality via local search
    improved = tools['apply_local_swap_in_out'](best_selection, time_limit_s=max(0.1, time_limit_s - (time.time() - start_time)))
    
    # Convert selection to binary list
    x = [0] * n
    for idx in improved:
        if 0 <= idx < n:
            x[idx] = 1
            
    return {'x': x}