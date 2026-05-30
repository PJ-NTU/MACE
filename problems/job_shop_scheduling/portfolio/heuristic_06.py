# MACE evolved heuristic 06/10 for problem: job_shop_scheduling
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Simulated Annealing (SA) heuristic operating directly on the 
    disjunctive graph representation's machine sequences.
    
    Fixed the TypeError by ensuring that the initial sequence generation
    and the evaluation logic handle edge cases properly, and providing
    a fallback if the simulation fails.
    """
    start_wall = time.time()
    n_jobs = tools['n_jobs']()
    n_machines = tools['n_machines']()
    
    # Ensure we start with a known feasible solution
    best_st = tools['spt_dispatch']()
    best_mks = max(tools['job_completion_time'](j, best_st) for j in range(n_jobs))
    
    # Helper to convert start_times back to sequences for the local search
    def get_sequences_from_st(st):
        seqs = {m: [] for m in range(1, n_machines + 1)}
        for j in range(n_jobs):
            for op in range(n_machines):
                m = tools['machine_of'](j, op)
                seqs[m].append((st[j][op], j))
        for m in seqs:
            seqs[m] = [job for _, job in sorted(seqs[m])]
        return seqs

    current_seqs = get_sequences_from_st(best_st)
    current_mks, current_st = best_mks, best_st
    
    # SA Parameters
    temp = 100.0
    cooling_rate = 0.9995
    
    while time.time() - start_wall < time_limit_s * 0.95:
        m = random.randint(1, n_machines)
        if len(current_seqs[m]) < 2:
            continue
            
        idx = random.randint(0, len(current_seqs[m]) - 2)
        
        # Swap
        current_seqs[m][idx], current_seqs[m][idx+1] = current_seqs[m][idx+1], current_seqs[m][idx]
        
        try:
            new_st = tools['simulate_active_schedule'](current_seqs)
            new_mks = max(tools['job_completion_time'](j, new_st) for j in range(n_jobs))
        except Exception:
            # Revert if simulation fails (deadlock/invalid sequence)
            current_seqs[m][idx], current_seqs[m][idx+1] = current_seqs[m][idx+1], current_seqs[m][idx]
            continue
        
        # Acceptance criterion
        delta = new_mks - current_mks
        if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
            current_mks, current_st = new_mks, new_st
            if current_mks < best_mks:
                best_mks, best_st = current_mks, current_st
        else:
            # Revert
            current_seqs[m][idx], current_seqs[m][idx+1] = current_seqs[m][idx+1], current_seqs[m][idx]
            
        # Cool down
        temp *= cooling_rate
        if temp < 0.1:
            temp = 10.0
            idx_m = random.randint(1, n_machines)
            random.shuffle(current_seqs[idx_m])
            
    return {"start_times": best_st}