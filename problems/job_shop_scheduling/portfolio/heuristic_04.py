# MACE evolved heuristic 04/10 for problem: job_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Synthesized solver for JSSP using instance-feature-based strategy selection.
    
    Strategy:
    - Small instances (total_ops <= 100) benefit from Tabu Search on machine sequences,
      which explores the disjunctive graph space more effectively than constructive methods.
    - Larger, sparser, or complex instances benefit from Regret-biased constructive
      multi-start + critical path refinement, which scales better and prevents
      getting lost in the search space.
    """
    start_wall = time.time()
    n_jobs = tools['n_jobs']()
    n_machines = tools['n_machines']()
    total_ops = n_jobs * n_machines
    
    # Feature calculation: Density/Bottleneck potential
    # If the variance in processing times is high, constructive heuristics with 
    # look-ahead (B) are generally more robust. If variance is low, local search (A)
    # on machine sequences is superior.
    times_flat = [t for row in instance['times'] for t in row]
    std_dev_times = sum((x - sum(times_flat)/total_ops)**2 for x in times_flat) / total_ops
    
    # Decision logic:
    # Use Tabu Machine Sequence Search (A-style) for small or uniform instances.
    # Use Regret-Biased Constructive Refinement (B-style) for larger or highly variable instances.
    if total_ops <= 80 or (total_ops <= 150 and std_dev_times < 100):
        return _solve_tabu_machine_sequence(instance, tools, time_limit_s)
    else:
        return _solve_regret_constructive(instance, tools, time_limit_s)

def _solve_tabu_machine_sequence(instance, tools, time_limit_s):
    start_wall = time.time()
    n_jobs = tools['n_jobs']()
    n_machines = tools['n_machines']()
    
    def get_initial_sequences():
        st = tools['spt_dispatch']()
        machine_ops = {m: [] for m in range(1, n_machines + 1)}
        for j in range(n_jobs):
            for op in range(n_machines):
                m = tools['machine_of'](j, op)
                machine_ops[m].append((st[j][op], j))
        seqs = {}
        for m in machine_ops:
            machine_ops[m].sort()
            seqs[m] = [x[1] for x in machine_ops[m]]
        return seqs

    sequences = get_initial_sequences()
    
    def evaluate(seqs):
        try:
            st = tools['simulate_active_schedule'](seqs)
            mks = max(tools['job_completion_time'](j, st) for j in range(n_jobs))
            return mks, st
        except:
            return float('inf'), None

    best_mks, best_st = evaluate(sequences)
    tabu_list = {}
    iters = 0
    while time.time() - start_wall < time_limit_s * 0.9:
        iters += 1
        m = random.randint(1, n_machines)
        if len(sequences[m]) < 2: continue
        idx = random.randint(0, len(sequences[m]) - 2)
        j1, j2 = sequences[m][idx], sequences[m][idx+1]
        if tabu_list.get((m, j1, j2), 0) > iters: continue
        sequences[m][idx], sequences[m][idx+1] = j2, j1
        mks, st = evaluate(sequences)
        if mks < best_mks:
            best_mks, best_st = mks, st
            tabu_list[(m, j1, j2)] = iters + 7
        else:
            sequences[m][idx], sequences[m][idx+1] = j1, j2
    return {"start_times": best_st}

def _solve_regret_constructive(instance, tools, time_limit_s):
    start_wall = time.time()
    n_jobs = tools['n_jobs']()
    n_machines = tools['n_machines']()
    size = n_jobs * n_machines
    
    def get_makespan(st):
        return max(tools['job_completion_time'](j, st) for j in range(n_jobs))

    best_st = tools['spt_dispatch']()
    best_mks = get_makespan(best_st)
    
    while time.time() - start_wall < time_limit_s * 0.8:
        current_st = [[0] * n_machines for _ in range(n_jobs)]
        job_next_op, m_free, j_free = [0] * n_jobs, {m: 0 for m in range(1, n_machines + 1)}, [0] * n_jobs
        for _ in range(size):
            ready = []
            for j in range(n_jobs):
                if job_next_op[j] < n_machines:
                    m = tools['machine_of'](j, job_next_op[j])
                    est = max(j_free[j], m_free[m])
                    ready.append((est, j, m))
            if not ready: break
            ready.sort()
            choice = ready[random.randrange(min(3, len(ready))) if random.random() > 0.1 else 0]
            st, j, m = choice
            current_st[j][job_next_op[j]] = st
            finish = st + tools['processing_time'](j, job_next_op[j])
            j_free[j] = m_free[m] = finish
            job_next_op[j] += 1
        mks = get_makespan(current_st)
        if mks < best_mks:
            best_mks, best_st = mks, current_st
            
    improved = tools['apply_critical_path_swap'](best_st, time_limit_s=max(0.1, time_limit_s * 0.1))
    return {"start_times": improved if improved else best_st}