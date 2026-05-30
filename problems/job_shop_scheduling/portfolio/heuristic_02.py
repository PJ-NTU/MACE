# MACE evolved heuristic 02/10 for problem: job_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Synthesized JSSP solver:
    1. Warm start using SPT dispatch.
    2. Multi-start construction using a randomized 'Priority' rule to explore
       diverse regions of the search space.
    3. Aggressive iterative local search using critical path swaps.
    """
    start_wall = time.time()
    n_jobs = tools['n_jobs']()
    n_machines = tools['n_machines']()
    times = instance['times']
    machines = instance['machines']

    def get_makespan(start_times):
        return max(tools['job_completion_time'](j, start_times) for j in range(n_jobs))

    # Best solution found
    best_start_times = tools['spt_dispatch']()
    best_makespan = get_makespan(best_start_times)

    # 1. Randomized Construction Phase
    # Uses a randomized priority rule: pick from candidates with high probability
    # of leading to smaller makespan based on remaining work and machine load.
    construction_limit = start_wall + (time_limit_s * 0.3)
    while time.time() < construction_limit:
        current_start_times = [[0] * n_machines for _ in range(n_jobs)]
        job_next_op = [0] * n_jobs
        machine_free_until = {m: 0 for m in range(1, n_machines + 1)}
        job_free_until = [0] * n_jobs
        
        for _ in range(n_jobs * n_machines):
            ready = []
            for j in range(n_jobs):
                if job_next_op[j] < n_machines:
                    m = machines[j][job_next_op[j]]
                    pt = times[j][job_next_op[j]]
                    est = max(job_free_until[j], machine_free_until[m])
                    # Priority: Earliest potential finish
                    ready.append((est + pt, j, m, pt, est))
            
            if not ready:
                break
            
            # Stochastic selection: biased towards lower EST
            ready.sort()
            # Pick from top 2 or 3 to maintain feasibility while exploring
            k = min(len(ready), 3)
            idx = random.randrange(k)
            _, j, m, pt, est = ready[idx]
            
            current_start_times[j][job_next_op[j]] = est
            finish = est + pt
            job_free_until[j] = finish
            machine_free_until[m] = finish
            job_next_op[j] += 1
            
        mks = get_makespan(current_start_times)
        if mks < best_makespan:
            best_makespan = mks
            best_start_times = current_start_times

    # 2. Refinement Phase
    # Use the provided neighborhood search tool as an engine.
    refine_end = start_wall + (time_limit_s * 0.95)
    
    # We iterate until time runs out, using the tool to jump to better local optima
    while time.time() < refine_end:
        # We pass a small time slice to the tool to remain reactive
        improved = tools['apply_critical_path_swap'](best_start_times, time_limit_s=0.08)
        if improved is not None:
            new_mks = get_makespan(improved)
            if new_mks < best_makespan:
                best_makespan = new_mks
                best_start_times = improved
            else:
                # No improvement found in this neighborhood; break to allow restart
                break
        else:
            break
            
    return {"start_times": best_start_times}