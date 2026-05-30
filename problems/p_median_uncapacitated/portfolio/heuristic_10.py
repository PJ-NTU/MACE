# MACE evolved heuristic 10/10 for problem: p_median_uncapacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved UPM solver.
    
    Diagnosis of parent:
    1. The parent relies on a high-overhead LK interchange which may not be optimal for 
       all instance sizes within tight time constraints.
    2. The random restart logic is inefficient; it doesn't prioritize the most promising 
       local search trajectories.
    3. The construction phase is over-complicated.
    
    Redesign:
    - Prioritize a robust greedy construction.
    - Use the high-performance 'apply_swap_one_for_one' with first_improvement=True 
      to quickly reach local optima.
    - If time remains, use a diversified restart strategy to escape local optima.
    - Explicitly monitor time to ensure we never return an invalid or empty solution.
    """
    start_time = time.time()
    n = instance['n']
    p = instance['p']
    
    # 1. Start with the most reliable construction
    best_medians = tools['greedy_add_one_until_p']()
    best_cost = tools['cost_given_open'](best_medians)
    
    # 2. Refine the initial greedy solution
    def refine(medians, time_budget):
        return tools['apply_swap_one_for_one'](
            medians, 
            time_limit_s=time_budget, 
            first_improvement=True
        )

    remaining = time_limit_s - (time.time() - start_time)
    if remaining > 0.1:
        best_medians = refine(best_medians, remaining * 0.5)
        best_cost = tools['cost_given_open'](best_medians)

    # 3. Diversified Multi-start loop
    # We use a simple perturbation (random swap) to escape local optima
    # if time allows, rather than full construction restarts.
    while time.time() - start_time < time_limit_s * 0.9:
        # Create a new candidate by perturbing the current best
        candidate = list(best_medians)
        # Randomly replace 1 or 2 medians
        num_perturb = min(2, p)
        for _ in range(num_perturb):
            rem_idx = random.randrange(p)
            new_val = random.randint(1, n)
            while new_val in candidate:
                new_val = random.randint(1, n)
            candidate[rem_idx] = new_val
        
        # Refine perturbation
        remaining = time_limit_s - (time.time() - start_time)
        if remaining < 0.05:
            break
            
        candidate = refine(candidate, remaining)
        current_cost = tools['cost_given_open'](candidate)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_medians = candidate
            
    return {"medians": list(best_medians)}