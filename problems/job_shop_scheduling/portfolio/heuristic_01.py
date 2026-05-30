# MACE evolved heuristic 01/10 for problem: job_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    JSSP solver that dispatches between a Boltzmann-based stochastic construction (A)
    and a Regret-biased greedy construction (B) based on problem density/size.
    
    Hypothesis: 
    - Smaller, tighter instances (n_jobs * n_machines < 100) benefit from the
      aggressive exploration of the Boltzmann-weighted selection (A).
    - Larger, more complex instances benefit from the structured pruning of the 
      Regret-biased greedy approach (B), which maintains better stability in 
      the search space.
    """
    start_wall = time.time()
    n_jobs = tools['n_jobs']()
    n_machines = tools['n_machines']()
    total_ops = n_jobs * n_machines
    
    # Heuristic dispatch: 
    # Use A (Boltzmann) for smaller/sparse, B (Regret) for large/dense.
    use_strategy_a = total_ops < 100
    
    def get_makespan(start_times):
        return max(tools['job_completion_time'](j, start_times) for j in range(n_jobs))

    best_start_times = tools['spt_dispatch']()
    best_makespan = get_makespan(best_start_times)
    
    construction_end = start_wall + (time_limit_s * 0.4)
    
    while time.time() < construction_end:
        current_start_times = [[0] * n_machines for _ in range(n_jobs)]
        job_next_op = [0] * n_jobs
        machine_free_until = {m: 0 for m in range(1, n_machines + 1)}
        job_free_until = [0] * n_jobs
        
        if use_strategy_a:
            # Boltzmann-weighted construction
            job_usage_count = [0] * n_jobs
            elapsed = (time.time() - start_wall) / (time_limit_s * 0.4 + 1e-9)
            temp = max(0.1, 1.0 - elapsed)
            
            for _ in range(total_ops):
                ready = []
                for j in range(n_jobs):
                    if job_next_op[j] < n_machines:
                        m = tools['machine_of'](j, job_next_op[j])
                        pt = tools['processing_time'](j, job_next_op[j])
                        ready.append((j, m, pt))
                if not ready: break
                
                pts = [r[2] for r in ready]
                penalty = [1.0 + (job_usage_count[r[0]] * 2.0) for r in ready]
                probs = [pow(2.718, -p / (temp * max(pts) + 1e-9)) / pen for p, pen in zip(pts, penalty)]
                sum_probs = sum(probs)
                choice_idx = random.choices(range(len(ready)), weights=[p/sum_probs for p in probs], k=1)[0]
                j, m, pt = ready[choice_idx]
                job_usage_count[j] += 1
                st = max(job_free_until[j], machine_free_until[m])
                current_start_times[j][job_next_op[j]] = st
                finish = st + pt
                job_free_until[j], machine_free_until[m], job_next_op[j] = finish, finish, job_next_op[j] + 1
        else:
            # Regret-biased greedy construction
            progress = (time.time() - start_wall) / (construction_end - start_wall + 1e-9)
            limit_size = max(1, int(5 * (1 - progress)))
            for _ in range(total_ops):
                ready = []
                for j in range(n_jobs):
                    if job_next_op[j] < n_machines:
                        m = tools['machine_of'](j, job_next_op[j])
                        ready.append({'j': j, 'm': m, 'pt': tools['processing_time'](j, job_next_op[j]), 
                                      'est': max(job_free_until[j], machine_free_until[m])})
                if not ready: break
                ready.sort(key=lambda x: x['est'])
                op = ready[random.randrange(min(limit_size, len(ready)))]
                current_start_times[op['j']][job_next_op[op['j']]] = op['est']
                finish = op['est'] + op['pt']
                job_free_until[op['j']] = machine_free_until[op['m']] = finish
                job_next_op[op['j']] += 1
            
        mks = get_makespan(current_start_times)
        if mks < best_makespan:
            best_makespan, best_start_times = mks, current_start_times

    refine_end = start_wall + (time_limit_s * 0.95)
    while time.time() < refine_end:
        improved = tools['apply_critical_path_swap'](best_start_times, time_limit_s=0.05)
        if improved is not None:
            new_mks = get_makespan(improved)
            if new_mks < best_makespan:
                best_makespan, best_start_times = new_mks, improved
            elif random.random() < 0.05: break
        else: break
            
    return {"start_times": best_start_times}