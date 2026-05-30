# MACE evolved heuristic 03/10 for problem: common_due_date_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for Common Due Date Scheduling.
    
    Hypothesis:
    - Small instances (n <= 30) benefit from the exhaustive, high-quality
      local search provided by the V-shape + 2-opt/Or-opt pipeline (Parent A).
    - Large instances (n > 30) require a more randomized, exploration-heavy 
      approach like the randomized insertion search (Parent B) to avoid 
      getting trapped in deep local minima within the time limit.
    """
    start_time = time.time()
    n = tools['n_jobs']()
    
    # Heuristic Dispatch:
    # If the problem size is small, prioritize high-quality local refinement.
    # If the problem size is large, prioritize randomized exploration.
    if n <= 30:
        # Parent A implementation
        best_schedule = tools['v_shape_construct']()
        best_penalty = tools['compute_total_penalty'](best_schedule)
        
        time_budget_for_search = time_limit_s * 0.9
        
        while time.time() - start_time < time_budget_for_search:
            current_best = best_penalty
            
            # Use provided optimized search tools
            improved_schedule = tools['apply_insertion_search'](
                best_schedule, 
                time_limit_s=max(0.1, (time_budget_for_search - (time.time() - start_time)) / 2)
            )
            
            final_schedule = tools['apply_swap_2opt'](
                improved_schedule, 
                time_limit_s=max(0.1, (time_budget_for_search - (time.time() - start_time)))
            )
            
            new_penalty = tools['compute_total_penalty'](final_schedule)
            if new_penalty < best_penalty:
                best_penalty = new_penalty
                best_schedule = final_schedule
            
            if best_penalty >= current_best:
                break
    else:
        # Parent B implementation
        v_sched = tools['v_shape_construct']()
        wspt_sched = tools['wspt_construct']()
        
        p_v = tools['compute_total_penalty'](v_sched)
        p_w = tools['compute_total_penalty'](wspt_sched)
        
        best_schedule = list(v_sched if p_v < p_w else wspt_sched)
        best_penalty = min(p_v, p_w)
        
        # Randomized local search for larger state spaces
        while (time.time() - start_time) < (time_limit_s - 0.1):
            i = random.randrange(n)
            j = random.randrange(n)
            if i == j:
                continue
                
            new_schedule = list(best_schedule)
            job = new_schedule.pop(i)
            new_schedule.insert(j, job)
            
            new_penalty = tools['compute_total_penalty'](new_schedule)
            if new_penalty < best_penalty:
                best_penalty = new_penalty
                best_schedule = new_schedule
                
    return {'schedule': list(best_schedule)}