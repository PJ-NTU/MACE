# MACE evolved heuristic 03/10 for problem: open_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Synthesized heuristic for Open Shop Scheduling.
    
    Dispatch Logic:
    - If n_jobs * n_machines is small (<= 49), use an exact ILP approach 
      as the primary driver, as the search space is highly constrained.
    - If the problem is mid-to-large, use a hybrid GRASP-LocalSearch strategy:
      - For "square" instances (n_jobs ≈ n_machines), use the LPT-construction
        as a structural anchor followed by intensified local swaps.
      - For "rectangular" instances (large imbalance), use randomized GRASP 
        priorities to explore the search space more widely.
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    total_ops = n_jobs * n_machines
    
    # Feature: Aspect Ratio
    aspect_ratio = max(n_jobs, n_machines) / min(n_jobs, n_machines)
    
    def get_makespan(starts):
        res = tools['simulate_makespan_from_starts'](starts)
        return float('inf') if res is None else res

    # 1. Exact Regimen (Small instances)
    if total_ops <= 49:
        ilp_time = min(time_limit_s * 0.6, 15.0)
        best_starts = tools['ilp_open_shop'](time_limit_s=ilp_time)
        if best_starts is None:
            best_starts = tools['lpt_dense_construct']()
        
        remaining = time_limit_s - (time.time() - start_time)
        if remaining > 0.1:
            best_starts = tools['apply_local_swap'](best_starts, time_limit_s=remaining)
            
    # 2. Square Regimen (Intensification)
    elif aspect_ratio < 1.5:
        best_starts = tools['lpt_dense_construct']()
        best_makespan = get_makespan(best_starts)
        
        # Iterative improvement on the LPT baseline
        while time.time() - start_time < time_limit_s * 0.9:
            remaining = time_limit_s - (time.time() - start_time)
            if remaining < 0.2: break
            
            # Apply local swap directly to the structure
            prev_starts = best_starts
            best_starts = tools['apply_local_swap'](prev_starts, time_limit_s=min(remaining, 5.0))
            if get_makespan(best_starts) >= best_makespan:
                break
            best_makespan = get_makespan(best_starts)
            
    # 3. Rectangular Regimen (Diversification via GRASP)
    else:
        best_starts = tools['lpt_dense_construct']()
        best_makespan = get_makespan(best_starts)
        
        while time.time() - start_time < time_limit_s * 0.85:
            ops = [(j, instance['machines'][j][m]) for j in range(n_jobs) for m in range(n_machines)]
            random.shuffle(ops)
            
            candidate_starts = tools['greedy_list_schedule'](priorities=ops)
            remaining = time_limit_s - (time.time() - start_time)
            
            if remaining > 0.2:
                refined = tools['apply_local_swap'](candidate_starts, time_limit_s=min(remaining, 2.0))
                c_makespan = get_makespan(refined)
                if c_makespan < best_makespan:
                    best_makespan = c_makespan
                    best_starts = refined
            else:
                break
                
    return {"start_times": best_starts}