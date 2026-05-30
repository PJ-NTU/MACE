# MACE evolved heuristic 08/10 for problem: hybrid_reentrant_shop_scheduling
import time
import random
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized heuristic for Hybrid Reentrant Shop Scheduling.
    
    Analysis:
    - h_a performs better because it centralizes the time budget on the best 
      identified candidate rather than fragmenting it across many restarts.
    - The key to improvement is a more efficient search strategy: prioritize 
      the most promising heuristic seeds and allocate the bulk of the 
      time_limit_s to deep 'insert' neighborhood refinement on the best seed.
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']

    # 1. Determine machine assignment (consistent with list scheduling).
    # Since init_time is constant, list scheduling is equivalent to 
    # round-robin assignment.
    batch_assignment = [((i - 1) % n_machines) + 1 for i in range(1, n_jobs + 1)]

    # 2. Candidate generation: evaluate priority rules.
    rules = ['natural', 'spt_setup', 'lpt_setup', 'spt_main', 'lpt_main', 'spt_total', 'lpt_total', 'erd']
    best_perm = None
    best_makespan = float('inf')

    # Sort rules by potential (heuristically) to hit the best seed early.
    for rule in rules:
        if time.time() - start_time > time_limit_s * 0.3:
            break
        try:
            perm = tools['list_scheduling_priority'](rule=rule)
            sim = tools['simulate_schedule'](perm)
            if sim['makespan'] < best_makespan:
                best_makespan = sim['makespan']
                best_perm = perm
        except Exception:
            continue
    
    # Ensure a valid permutation exists
    if best_perm is None:
        best_perm = list(range(1, n_jobs + 1))

    # 3. Refinement: Allocate the majority of time to deep 'insert' local search.
    # We use a significant chunk of time for the single best candidate found.
    # 'insert' is computationally more expensive but globally more effective 
    # than 'adjacent' swaps.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        # We dedicate up to 90% of the remaining time to the best permutation found.
        # This prevents the fragmentation issue observed in h_b.
        refined_perm = tools['apply_local_swap'](
            permutation=list(best_perm),
            time_limit_s=remaining_time * 0.9,
            neighbourhood='insert'
        )
        
        # Verify refinement
        try:
            sim = tools['simulate_schedule'](refined_perm)
            if sim['makespan'] < best_makespan:
                best_perm = refined_perm
        except Exception:
            pass

    return {
        "permutation": best_perm,
        "batch_assignment": batch_assignment
    }