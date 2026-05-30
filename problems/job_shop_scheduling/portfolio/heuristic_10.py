# MACE evolved heuristic 10/10 for problem: job_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized JSSP solver utilizing a robust multi-start framework:
    1. Warm start using ILP (if small) or SPT (if large).
    2. Iterative local search using critical path swaps.
    3. Stochastic greedy construction with adaptive selection pressure.
    """
    start_wall = time.time()
    n_jobs = tools['n_jobs']()
    n_machines = tools['n_machines']()
    total_ops = n_jobs * n_machines
    
    def get_makespan(st):
        return max(tools['job_completion_time'](j, st) for j in range(n_jobs))

    # 1. Initialization
    best_st = None
    if total_ops <= 100 and time_limit_s > 1.0:
        best_st = tools['ilp_jssp'](time_limit_s=min(time_limit_s * 0.3, 5.0))
    
    if best_st is None:
        best_st = tools['spt_dispatch']()
        
    best_mks = get_makespan(best_st)
    
    # 2. Refinement Loop
    # We balance between local search (intensive improvement) and 
    # constructive restarts (exploration).
    
    while time.time() - start_wall < time_limit_s * 0.9:
        # Phase A: Local Search (Critical Path)
        # Apply critical path swaps to refine the current best solution
        improved = tools['apply_critical_path_swap'](best_st, time_limit_s=max(0.05, time_limit_s * 0.05))
        if improved:
            imp_mks = get_makespan(improved)
            if imp_mks < best_mks:
                best_mks, best_st = imp_mks, improved
        
        # Phase B: Stochastic Construction (Restart)
        # Use a "Regret-biased" or "Stochastic Greedy" approach to explore the space
        elapsed_ratio = (time.time() - start_wall) / time_limit_s
        # Increase randomness early, focus on refinement late
        k = max(1, int(5 * (1.0 - elapsed_ratio)))
        
        current_st = [[0] * n_machines for _ in range(n_jobs)]
        job_next, m_free, j_free = [0] * n_jobs, {m: 0 for m in range(1, n_machines + 1)}, [0] * n_jobs
        
        for _ in range(total_ops):
            ready = []
            for j in range(n_jobs):
                if job_next[j] < n_machines:
                    m = tools['machine_of'](j, job_next[j])
                    est = max(j_free[j], m_free[m])
                    ready.append((est, j, m, tools['processing_time'](j, job_next[j])))
            
            if not ready: break
            
            # Sort by earliest start time (EST)
            ready.sort(key=lambda x: x[0])
            
            # Stochastic choice: pick from top k to maintain diversity
            idx = random.randrange(min(k, len(ready)))
            est, j, m, pt = ready[idx]
            
            current_st[j][job_next[j]] = est
            finish = est + pt
            j_free[j] = m_free[m] = finish
            job_next[j] += 1
            
        c_mks = get_makespan(current_st)
        if c_mks < best_mks:
            best_mks, best_st = c_mks, current_st
            
    return {"start_times": best_st}