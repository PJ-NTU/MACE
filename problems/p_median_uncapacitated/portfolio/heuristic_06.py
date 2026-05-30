# MACE evolved heuristic 06/10 for problem: p_median_uncapacitated
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher heuristic for the Uncapacitated P-Median Problem.
    
    Hypothesis:
    - Deterministic, greedy-based local search (Teitz-Bart) is highly efficient 
      for smaller instances (n < 100) or instances where the cost landscape is 
      relatively smooth/convex.
    - Simulated Annealing is superior for larger, more complex, or highly 
      non-convex landscapes where random restarts and stochastic exploration 
      can escape deep, sub-optimal local traps.
    """
    start_time = time.time()
    n = instance['n']
    p = instance['p']

    # Dispatch Logic:
    # Use deterministic Teitz-Bart for smaller/dense problems where 
    # greedy-local search is likely to find the global optimum quickly.
    # Use SA for larger problems where the search space is too vast for 
    # simple hill-climbing to avoid poor local optima.
    if n <= 100:
        # Strategy A: Greedy + Teitz-Bart
        try:
            current_medians = tools['greedy_add_one_until_p']()
        except:
            current_medians = random.sample(range(1, n + 1), p)
            
        remaining_time = time_limit_s - (time.time() - start_time)
        if remaining_time > 0.1:
            current_medians = tools['apply_swap_one_for_one'](
                current_medians, 
                time_limit_s=remaining_time, 
                first_improvement=True
            )
        return {"medians": list(current_medians)}
    
    else:
        # Strategy B: Simulated Annealing
        # We use a more aggressive cooling schedule for larger instances.
        current_medians = random.sample(range(1, n + 1), p)
        current_cost = tools['cost_given_open'](current_medians)
        
        best_medians = list(current_medians)
        best_cost = current_cost
        
        temp = 100.0
        cooling_rate = 0.9999
        
        count = 0
        while time.time() - start_time < time_limit_s * 0.95:
            idx_to_remove = random.randrange(p)
            new_median = random.randint(1, n)
            while new_median in current_medians:
                new_median = random.randint(1, n)
            
            candidate = list(current_medians)
            candidate[idx_to_remove] = new_median
            
            candidate_cost = tools['cost_given_open'](candidate)
            delta = candidate_cost - current_cost
            
            if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
                current_medians = candidate
                current_cost = candidate_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_medians = list(current_medians)
            
            temp *= cooling_rate
            count += 1
            if count % 1000 == 0 and temp < 0.01:
                temp = 5.0 # Reheat
                
        return {"medians": list(best_medians)}