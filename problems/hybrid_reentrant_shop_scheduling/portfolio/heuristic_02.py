# MACE evolved heuristic 02/10 for problem: hybrid_reentrant_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Heuristic for Hybrid Reentrant Shop Scheduling:
    1. The 'batch_assignment' is deterministic (based on list scheduling) per the problem spec.
    2. The permutation is optimized via a combination of priority-rule warm starts 
       and a First-Improvement Local Search using the 'insert' neighborhood.
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']

    # The batch_assignment is fixed by the problem logic:
    # "Jobs are initialized in a fixed natural order using list scheduling...
    # and that assignment is used for the final main processing."
    # We reconstruct the machine mapping used by the simulator.
    batch_assignment = [0] * n_jobs
    machine_finish_times = [0] * n_machines
    init_time = instance['init_time']
    
    # Simulate list scheduling to get the required batch_assignment
    for j in range(n_jobs):
        # Find machine with earliest finish time
        m_idx = min(range(n_machines), key=lambda x: machine_finish_times[x])
        batch_assignment[j] = m_idx + 1
        machine_finish_times[m_idx] += init_time

    # Generate candidate permutations using priority rules
    rules = ['natural', 'spt_setup', 'lpt_setup', 'spt_main', 'lpt_main', 'spt_total', 'lpt_total', 'erd']
    best_perm = None
    best_makespan = float('inf')

    for rule in rules:
        if time.time() - start_time > time_limit_s * 0.2:
            break
        try:
            perm = tools['list_scheduling_priority'](rule=rule)
            sim = tools['simulate_schedule'](perm)
            if sim['makespan'] < best_makespan:
                best_makespan = sim['makespan']
                best_perm = perm
        except:
            continue

    if best_perm is None:
        best_perm = list(range(1, n_jobs + 1))

    # Refine with local search
    # We use 'insert' for better exploration, but monitor time carefully
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        refined_perm = tools['apply_local_swap'](
            best_perm, 
            time_limit_s=remaining_time, 
            neighbourhood='insert'
        )
        best_perm = refined_perm

    return {
        "permutation": best_perm,
        "batch_assignment": batch_assignment
    }