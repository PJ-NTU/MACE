# MACE evolved heuristic 09/10 for problem: multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized solver for the Multidimensional Knapsack Problem.
    
    Strategy:
    1. Robust Greedy Initialization: Compare density vs efficiency heuristics.
    2. Primary ILP Engine: Use the exact solver as the core driver, given its
       high performance on these constraints.
    3. Time-Aware Iterative Improvement: Instead of random LNS destruction,
       focus on a 'Deep Local Search' approach using the ILP solver to
       perform neighborhood refinements around the best-found solution.
    """
    start_time = time.time()
    n = tools['n_items']()
    
    # 1. Greedy Initialization
    greedy_density = tools['greedy_by_profit_density']()
    greedy_efficiency = tools['greedy_by_efficiency']()
    
    best_selection = greedy_density if tools['profit_of_selection'](greedy_density) > \
                                       tools['profit_of_selection'](greedy_efficiency) else greedy_efficiency
    best_profit = tools['profit_of_selection'](best_selection)
    
    # 2. Primary ILP Solve
    # Allocate the majority of time to the exact solver which is often optimal.
    # If the instance is small enough, this finishes quickly.
    # If large, it provides the best possible foundation.
    ilp_time = max(1.0, time_limit_s * 0.7)
    ilp_res = tools['ilp_solve_mkp'](time_limit_s=ilp_time)
    
    if ilp_res is not None:
        p = tools['profit_of_selection'](ilp_res)
        if p > best_profit:
            best_selection = ilp_res
            best_profit = p
            
    # 3. Iterative Neighborhood Search (Fix-and-Optimize)
    # Instead of random destruction, we perform a deterministic 'Fix-and-Optimize'
    # round if time allows. We fix the current best and try to improve small windows.
    # We use the ILP solver to optimize a subset of variables while keeping others fixed.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 1.0:
        # Sort items by profit/resource ratio to identify 'important' items to keep fixed
        # or flip. Here we use a simple strategy: fix the best, allow ILP to swap 
        # a subset of the remaining items.
        
        # We only attempt this if we have significant time left.
        # Fix 80% of current selection, allow ILP to optimize the remaining 20%.
        current_selection = set(best_selection)
        must_include = random.sample(list(current_selection), min(len(current_selection), int(0.8 * len(current_selection))))
        
        refined_res = tools['ilp_solve_mkp'](
            must_include=must_include, 
            time_limit_s=min(remaining_time * 0.5, 5.0)
        )
        
        if refined_res is not None:
            p_new = tools['profit_of_selection'](refined_res)
            if p_new > best_profit:
                best_selection = refined_res
                best_profit = p_new

    # 4. Final Polish
    # Always perform a final greedy swap refinement to close any gaps.
    final_selection = tools['apply_local_swap_in_out'](
        best_selection, 
        time_limit_s=max(0.1, time_limit_s - (time.time() - start_time))
    )
    
    # Format output
    x = [0] * n
    for idx in final_selection:
        if 0 <= idx < n:
            x[idx] = 1
            
    return {'x': x}