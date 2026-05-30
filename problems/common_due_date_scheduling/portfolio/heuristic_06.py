# MACE evolved heuristic 06/10 for problem: common_due_date_scheduling
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Synthesized hybrid solver:
    - Uses the V-shape construction as the universal foundation.
    - For small instances (N <= 40), uses deterministic, aggressive local search 
      (insertion + 2-opt) to reach local optima efficiently.
    - For large instances (N > 40), uses Simulated Annealing to navigate the 
      larger combinatorial landscape, leveraging the V-shape as a high-quality 
      starting point to avoid early entrapment.
    """
    start_time = time.time()
    n = tools['n_jobs']()
    
    # 1. Warm start: V-shape is optimal for the unconstrained Common Due Date problem
    best_schedule = tools['v_shape_construct']()
    best_penalty = tools['compute_total_penalty'](best_schedule)
    
    # 2. Heuristic Dispatch Logic
    # Small N: High-intensity local search (Or-opt + 2-opt)
    # Large N: Simulated Annealing to escape local minima
    if n <= 40:
        # Aggressive refinement for smaller spaces
        remaining_time = time_limit_s * 0.9
        
        while time.time() - start_time < remaining_time:
            # Use Or-opt (insertion) to move jobs across the due-date threshold
            improved = tools['apply_insertion_search'](
                best_schedule, 
                time_limit_s=max(0.1, (remaining_time - (time.time() - start_time)) / 2)
            )
            # Use 2-opt to refine local sequences
            final = tools['apply_swap_2opt'](
                improved, 
                time_limit_s=max(0.1, (remaining_time - (time.time() - start_time)))
            )
            
            p = tools['compute_total_penalty'](final)
            if p < best_penalty:
                best_penalty = p
                best_schedule = final
            else:
                break
    else:
        # Simulated Annealing for large spaces
        current_schedule = list(best_schedule)
        current_penalty = best_penalty
        
        # Scale initial temperature based on average penalty magnitude
        temp = best_penalty * 0.1
        cooling_rate = 0.9999
        
        while time.time() - start_time < time_limit_s * 0.95:
            # Neighbor: random swap
            i, j = random.sample(range(n), 2)
            new_schedule = list(current_schedule)
            new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
            
            new_penalty = tools['compute_total_penalty'](new_schedule)
            
            delta = new_penalty - current_penalty
            if delta < 0 or (temp > 1e-6 and random.random() < math.exp(-delta / temp)):
                current_schedule = new_schedule
                current_penalty = new_penalty
                
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_schedule = list(current_schedule)
            
            temp *= cooling_rate
            
            # Re-heat if frozen
            if temp < 1e-4:
                temp = best_penalty * 0.05
                
    return {'schedule': list(best_schedule)}