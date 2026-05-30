# MACE evolved heuristic 09/10 for problem: job_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized JSSP solver using a hybrid approach:
    1. Multi-start constructive search (randomized priority rules).
    2. Local search via critical path block swaps.
    3. Adaptive time management to maximize exploration within the budget.
    """
    start_time = time.time()
    n_jobs = tools['n_jobs']()
    n_machines = tools['n_machines']()
    
    def get_makespan(st):
        return max(tools['job_completion_time'](j, st) for j in range(n_jobs))

    # Helper: Randomized priority dispatching
    def construct_solution(randomness=0.2):
        st = [[0] * n_machines for _ in range(n_jobs)]
        job_next_op = [0] * n_jobs
        machine_free = {m: 0 for m in range(1, n_machines + 1)}
        job_free = [0] * n_jobs
        
        for _ in range(n_jobs * n_machines):
            candidates = []
            for j in range(n_jobs):
                if job_next_op[j] < n_machines:
                    m = tools['machine_of'](j, job_next_op[j])
                    est = max(job_free[j], machine_free[m])
                    # Priority based on earliest start + remaining work (bottleneck heuristic)
                    priority = est + 0.1 * tools['total_work'](j)
                    candidates.append((priority, j, m))
            
            if not candidates: break
            candidates.sort()
            
            # Choose from top candidates with small probability of random choice
            idx = 0
            if random.random() < randomness:
                idx = random.randint(0, min(len(candidates) - 1, 2))
            
            _, j, m = candidates[idx]
            op = job_next_op[j]
            pt = tools['processing_time'](j, op)
            
            start_t = max(job_free[j], machine_free[m])
            st[j][op] = start_t
            
            finish = start_t + pt
            job_free[j] = finish
            machine_free[m] = finish
            job_next_op[j] += 1
        return st

    # Initialize
    best_st = tools['spt_dispatch']()
    best_makespan = get_makespan(best_st)
    
    # Time-budgeted search
    # We allocate 40% to construction, 50% to critical path refinement, 10% safety buffer
    while time.time() - start_time < time_limit_s * 0.9:
        # Periodically attempt fresh construction to escape local basins
        if random.random() < 0.3:
            candidate_st = construct_solution(randomness=0.3)
        else:
            # Or refine current best
            candidate_st = tools['apply_critical_path_swap'](best_st, time_limit_s=0.2)
            if not candidate_st:
                continue
        
        mks = get_makespan(candidate_st)
        if mks < best_makespan:
            best_makespan = mks
            best_st = candidate_st
        
        # Early exit if time is running out
        if time.time() - start_time > time_limit_s * 0.95:
            break
            
    return {"start_times": best_st}