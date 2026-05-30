# MACE evolved heuristic 04/10 for problem: multidimensional_knapsack_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for MKP.
    
    Hypothesis:
    - Small to Medium instances (n < 500) or highly constrained instances benefit 
      from the exact ILP solver (Parent B's strength).
    - Large, sparse, or loosely constrained instances (where n is large) benefit 
      more from LNS metaheuristics (Parent A's strength) as the ILP solver 
      might struggle to close the gap within the time limit.
    """
    start_time = time.time()
    n = instance['n']
    m = instance['m']
    
    # Feature calculation: Density of constraints
    # A low total sum of resources relative to capacities suggests loose constraints.
    # We use a simple proxy: average resource consumption of items vs capacity.
    avg_resource_usage = sum(sum(r_row) / n for r_row in instance['r'])
    avg_capacity = sum(instance['b']) / m
    tightness = avg_resource_usage / (avg_capacity + 1e-9)

    # Dispatch Logic:
    # ILP is powerful for n < 400 or very tight constraints.
    # LNS is better for large n or loose problem spaces.
    use_ilp_first = (n < 400) or (tightness > 0.5)

    best_selection = []
    best_profit = -1.0

    def get_profit(sel):
        return tools['profit_of_selection'](sel)

    if use_ilp_first:
        # Parent B approach: ILP priority
        try:
            ilp_result = tools['ilp_solve_mkp'](time_limit_s=max(1.0, time_limit_s * 0.6))
            if ilp_result is not None:
                best_selection = ilp_result
                best_profit = get_profit(best_selection)
        except:
            pass
        
        # Refine with local search if time remains
        if time.time() - start_time < time_limit_s * 0.9:
            refined = tools['apply_local_swap_in_out'](best_selection or [], time_limit_s=time_limit_s * 0.2)
            if get_profit(refined) > best_profit:
                best_selection = refined
    else:
        # Parent A approach: LNS priority
        candidates = [tools['greedy_by_profit_density'](), tools['greedy_by_efficiency']()]
        for seed in candidates:
            if time.time() - start_time > time_limit_s * 0.3: break
            improved = tools['apply_local_swap_in_out'](seed, time_limit_s=time_limit_s * 0.1)
            if get_profit(improved) > best_profit:
                best_profit = get_profit(improved)
                best_selection = improved
        
        # LNS loop
        while time.time() - start_time < time_limit_s * 0.8:
            if not best_selection: break
            num = random.randint(1, max(1, len(best_selection) // 5))
            subset = [best_selection[i] for i in range(len(best_selection)) if i not in random.sample(range(len(best_selection)), num)]
            repaired = tools['repair_capacity_violation'](subset)
            improved = tools['apply_local_swap_in_out'](repaired, time_limit_s=time_limit_s * 0.05)
            if get_profit(improved) > best_profit:
                best_profit = get_profit(improved)
                best_selection = improved

    # Final fallback
    if not best_selection:
        best_selection = tools['greedy_by_efficiency']()

    x = [0] * n
    for idx in best_selection:
        if 0 <= idx < n:
            x[idx] = 1
            
    return {'x': x}