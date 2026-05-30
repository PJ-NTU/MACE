# MACE evolved heuristic 03/10 for problem: job_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Hybrid JSSP solver.
    Regime analysis:
    - Small/Tight: ILP + Critical Path Search (for optimality in constrained spaces).
    - Large/Complex: Randomized Greedy Construction + Local Search (for exploration).
    """
    start_wall = time.time()
    n_jobs = tools['n_jobs']()
    n_machines = tools['n_machines']()
    size = n_jobs * n_machines
    
    # Feature extraction: JSSP hardness is often defined by the "shop ratio" 
    # and total operations. Small, dense instances are best handled by exact 
    # methods or high-intensity local search. Large, sparse instances favor
    # construction-based heuristics.
    is_small_and_constrained = (size <= 100)
    
    def get_makespan(st):
        return max(tools['job_completion_time'](j, st) for j in range(n_jobs))

    # Initial solution
    if is_small_and_constrained and time_limit_s > 2.0:
        best_st = tools['ilp_jssp'](time_limit_s=min(time_limit_s * 0.4, 10.0))
        if best_st is None:
            best_st = tools['spt_dispatch']()
    else:
        best_st = tools['spt_dispatch']()
    
    best_mks = get_makespan(best_st)
    
    # Optimization Loop
    # We use a threshold of 0.95 to ensure we leave time for final return
    while time.time() - start_wall < time_limit_s * 0.95:
        # Phase 1: Local Improvement via Critical Path
        improved = tools['apply_critical_path_swap'](best_st, time_limit_s=0.1)
        
        if improved is not None:
            imp_mks = get_makespan(improved)
            if imp_mks < best_mks:
                best_mks, best_st = imp_mks, improved
        
        # Phase 2: Escape local optima
        # If we are stuck (no improvement from critical path), perform a 
        # perturbation. For larger instances, use greedy construction to reset.
        if time.time() - start_wall > time_limit_s * 0.5:
            if not is_small_and_constrained:
                # Stochastic greedy construction to explore new regions
                current_st = [[0] * n_machines for _ in range(n_jobs)]
                job_next, m_free, j_free = [0] * n_jobs, {m: 0 for m in range(1, n_machines + 1)}, [0] * n_jobs
                for _ in range(size):
                    ready = []
                    for j in range(n_jobs):
                        if job_next[j] < n_machines:
                            m = tools['machine_of'](j, job_next[j])
                            ready.append((j, m, tools['processing_time'](j, job_next[j]), max(j_free[j], m_free[m])))
                    if not ready: break
                    # Select from top 2 to add stochasticity
                    ready.sort(key=lambda x: x[3])
                    choice = ready[random.randrange(min(2, len(ready)))]
                    j, m, pt, start = choice
                    current_st[j][job_next[j]] = start
                    finish = start + pt
                    j_free[j] = m_free[m] = finish
                    job_next[j] += 1
                
                c_mks = get_makespan(current_st)
                if c_mks < best_mks:
                    best_mks, best_st = c_mks, current_st
            else:
                # For small instances, simple random swap on the machine sequence
                # to shake the local optimum
                break 

    return {"start_times": best_st}