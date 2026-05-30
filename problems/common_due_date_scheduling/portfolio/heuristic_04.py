# MACE evolved heuristic 04/10 for problem: common_due_date_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for the Common Due Date Scheduling problem.
    
    Logic:
    - Small instances (n <= 30) are solved optimally using the ILP solver.
    - Medium-to-Large instances (n > 30) are handled by a hybrid strategy:
        - If the instance is small enough that local search is highly effective 
          (n <= 60), we use a structured V-shape initialization followed by 
          an intensive insertion-based local search.
        - For very large instances (n > 60), the state space is too vast for simple 
          local search to escape deep basins effectively. We use Simulated Annealing 
          (based on Parent A) to provide better exploration of the permutation space 
          given the limited time budget.
    """
    start_time = time.time()
    n = tools['n_jobs']()

    # Regime 1: Small instances - Exact ILP
    if n <= 30:
        ilp_sol = tools['ilp_cdd'](time_limit_s=max(1.0, time_limit_s * 0.7))
        if ilp_sol:
            return {'schedule': list(ilp_sol)}
        # Fallback to V-shape if ILP fails
        best_schedule = tools['v_shape_construct']()
    
    # Regime 2: Medium instances - Targeted Local Search (Parent B style)
    elif n <= 60:
        best_schedule = tools['v_shape_construct']()
        remaining_time = time_limit_s - (time.time() - start_time) - 0.2
        if remaining_time > 0.1:
            best_schedule = tools['apply_insertion_search'](
                best_schedule, 
                time_limit_s=remaining_time
            )
            
    # Regime 3: Large instances - Simulated Annealing (Parent A style)
    else:
        current_schedule = tools['v_shape_construct']()
        current_penalty = tools['compute_total_penalty'](current_schedule)
        best_schedule = list(current_schedule)
        best_penalty = current_penalty
        
        temp = 1000.0
        cooling_rate = 0.9999
        
        while time.time() - start_time < time_limit_s * 0.95:
            new_schedule = list(current_schedule)
            i, j = random.sample(range(n), 2)
            new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
            
            new_penalty = tools['compute_total_penalty'](new_schedule)
            delta = new_penalty - current_penalty
            
            if delta < 0 or (temp > 0.01 and random.random() < pow(2.718, -delta / temp)):
                current_schedule = new_schedule
                current_penalty = new_penalty
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_schedule = list(current_schedule)
            
            temp *= cooling_rate
            if temp < 0.1:
                temp = 100.0
                
    return {'schedule': list(best_schedule)}