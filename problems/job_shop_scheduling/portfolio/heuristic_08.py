# MACE evolved heuristic 08/10 for problem: job_shop_scheduling
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Enhanced JSSP solver combining multi-start greedy construction with
    probabilistic local search and critical-path refinement.
    """
    start_wall = time.time()
    n_jobs = tools['n_jobs']()
    n_machines = tools['n_machines']()
    total_ops = n_jobs * n_machines
    
    def get_makespan(st):
        return max(tools['job_completion_time'](j, st) for j in range(n_jobs))

    # Initial solution
    best_st = tools['spt_dispatch']()
    best_mks = get_makespan(best_st)
    
    # Adaptive construction phase
    # h_b's sampling (k=5) is useful, but h_a's narrower sampling (k=3) 
    # is more robust for small instances. We use a time-decaying k.
    construction_end = start_wall + (time_limit_s * 0.4)
    while time.time() < construction_end:
        current_st = [[0] * n_machines for _ in range(n_jobs)]
        job_next_op, machine_free = [0] * n_jobs, {m: 0 for m in range(1, n_machines + 1)}
        job_free = [0] * n_jobs
        
        # Use remaining work heuristic to guide greedy choices
        remaining_work = [sum(tools['processing_time'](j, op) for op in range(job_next_op[j], n_machines)) 
                          for j in range(n_jobs)]
        
        for _ in range(total_ops):
            ready = []
            for j in range(n_jobs):
                if job_next_op[j] < n_machines:
                    m = tools['machine_of'](j, job_next_op[j])
                    pt = tools['processing_time'](j, job_next_op[j])
                    est = max(job_free[j], machine_free[m])
                    # Priority: Earliest start time, tie-break with Most Remaining Work (MRW)
                    ready.append({'j': j, 'm': m, 'pt': pt, 'est': est, 'rem': remaining_work[j]})
            
            if not ready: break
            
            # Sort by EST, then by remaining work
            ready.sort(key=lambda x: (x['est'], -x['rem']))
            
            # Adaptive k: early in search explore more, later exploit
            k = max(2, min(5, int(5 * (1 - (time.time() - start_wall) / time_limit_s))))
            choice = ready[random.randrange(min(k, len(ready)))]
            
            current_st[choice['j']][job_next_op[choice['j']]] = choice['est']
            finish = choice['est'] + choice['pt']
            job_free[choice['j']], machine_free[choice['m']] = finish, finish
            job_next_op[choice['j']] += 1
            remaining_work[choice['j']] -= choice['pt']
            
        mks = get_makespan(current_st)
        if mks < best_mks:
            best_mks, best_st = mks, current_st

    # Refinement phase
    # Use critical path swapping to polish the best found construction
    refine_end = start_wall + (time_limit_s * 0.95)
    while time.time() < refine_end:
        # Apply critical path swaps with a slightly more generous budget
        improved = tools['apply_critical_path_swap'](best_st, time_limit_s=0.1)
        if improved is not None:
            new_mks = get_makespan(improved)
            if new_mks < best_mks:
                best_mks, best_st = new_mks, improved
            else:
                # Stochastic restart to escape local optima
                break
        else:
            break
            
    return {"start_times": best_st}