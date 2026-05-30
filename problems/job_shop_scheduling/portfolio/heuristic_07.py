# MACE evolved heuristic 07/10 for problem: job_shop_scheduling
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Hybrid JSSP solver combining:
    1. A robust Regret-Biased Greedy construction for initialization.
    2. A Multi-Start Local Search phase using critical path swaps.
    3. Time-aware adaptation of exploration depth.
    """
    start_wall = time.time()
    n_jobs = tools['n_jobs']()
    n_machines = tools['n_machines']()
    
    def get_makespan(st):
        return max(tools['job_completion_time'](j, st) for j in range(n_jobs))

    # Initialize with SPT dispatch
    best_start_times = tools['spt_dispatch']()
    best_makespan = get_makespan(best_start_times)
    
    # 1. Randomized Greedy Construction Phase
    # Uses a regret-based approach to explore diverse feasible regions
    construction_end = start_wall + (time_limit_s * 0.3)
    while time.time() < construction_end:
        current_start_times = [[0] * n_machines for _ in range(n_jobs)]
        job_next_op = [0] * n_jobs
        machine_free = {m: 0 for m in range(1, n_machines + 1)}
        job_free = [0] * n_jobs
        
        # Stochastic selection from top-k ready operations
        # K decreases over time to converge on exploitation
        k_val = max(2, int(n_jobs * 0.5 * (1 - (time.time() - start_wall) / (time_limit_s * 0.3))))
        
        for _ in range(n_jobs * n_machines):
            ready = []
            for j in range(n_jobs):
                if job_next_op[j] < n_machines:
                    m = tools['machine_of'](j, job_next_op[j])
                    pt = tools['processing_time'](j, job_next_op[j])
                    est = max(job_free[j], machine_free[m])
                    ready.append({'j': j, 'm': m, 'pt': pt, 'est': est})
            
            if not ready: break
            
            # Sort by earliest start time (EST)
            ready.sort(key=lambda x: x['est'])
            idx = random.randrange(min(k_val, len(ready)))
            choice = ready[idx]
            
            current_start_times[choice['j']][job_next_op[choice['j']]] = choice['est']
            finish = choice['est'] + choice['pt']
            job_free[choice['j']] = machine_free[choice['m']] = finish
            job_next_op[choice['j']] += 1
            
        mks = get_makespan(current_start_times)
        if mks < best_makespan:
            best_makespan, best_start_times = mks, current_start_times

    # 2. Critical Path Refinement Phase
    # Focuses on the bottleneck operations identified by the framework
    refine_end = start_wall + (time_limit_s * 0.95)
    while time.time() < refine_end:
        # Utilize the provided tool for targeted neighborhood search
        improved = tools['apply_critical_path_swap'](best_start_times, time_limit_s=0.05)
        if improved is not None:
            new_mks = get_makespan(improved)
            if new_mks < best_makespan:
                best_makespan, best_start_times = new_mks, improved
            else:
                # Occasional restart to escape local optima
                if random.random() < 0.02:
                    best_start_times = tools['spt_dispatch']()
        else:
            # If no improvement found, perform a small restart
            best_start_times = tools['spt_dispatch']()
            
    return {"start_times": best_start_times}