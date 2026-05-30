# MACE evolved heuristic 07/10 for problem: common_due_date_scheduling
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Synthesized hybrid solver:
    - Uses V-shape initialization as a common strong foundation.
    - For small instances (n <= 40), it utilizes aggressive local search (insertion/swap)
      to exploit the known structure of the problem.
    - For larger, more complex instances, it uses a Simulated Annealing schedule 
      to explore the search space, preventing convergence to sub-optimal local minima.
    """
    start_time = time.time()
    n = tools['n_jobs']()
    
    # Heuristic Decision:
    # Small n: The search space is small enough that greedy local search (hill climbing)
    # with multiple restarts or exhaustive passes is highly effective at finding the global optimum.
    # Large n: The landscape becomes more rugged; stochastic exploration (SA) is more robust.
    
    if n <= 40:
        # Strategy A: Iterative Local Search refinement on V-Shape
        best_schedule = tools['v_shape_construct']()
        best_penalty = tools['compute_total_penalty'](best_schedule)
        
        # Use a portion of the time to refine the V-shape baseline
        search_budget = time_limit_s * 0.95
        while time.time() - start_time < search_budget:
            # Or-opt is excellent for shifting jobs across the V-shape boundary
            improved = tools['apply_insertion_search'](
                best_schedule, 
                time_limit_s=max(0.1, (search_budget - (time.time() - start_time)) * 0.6)
            )
            # 2-opt fine-tunes the local ordering
            refined = tools['apply_swap_2opt'](
                improved, 
                time_limit_s=max(0.1, (search_budget - (time.time() - start_time)))
            )
            
            p_new = tools['compute_total_penalty'](refined)
            if p_new < best_penalty:
                best_penalty = p_new
                best_schedule = refined
            else:
                break
    else:
        # Strategy B: Simulated Annealing for larger/complex landscapes
        current_schedule = tools['v_shape_construct']()
        current_penalty = tools['compute_total_penalty'](current_schedule)
        best_schedule = list(current_schedule)
        best_penalty = current_penalty
        
        temp = 500.0
        cooling_rate = 0.9999
        
        while time.time() - start_time < time_limit_s * 0.95:
            # Perturb the schedule via a random 2-swap
            new_schedule = list(current_schedule)
            i, j = random.sample(range(n), 2)
            new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
            
            new_penalty = tools['compute_total_penalty'](new_schedule)
            delta = new_penalty - current_penalty
            
            # Acceptance logic
            if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
                current_schedule = new_schedule
                current_penalty = new_penalty
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_schedule = list(current_schedule)
            
            temp *= cooling_rate
            # Periodic re-heating to escape deep basins
            if temp < 0.05:
                temp = 100.0
                
    return {'schedule': list(best_schedule)}