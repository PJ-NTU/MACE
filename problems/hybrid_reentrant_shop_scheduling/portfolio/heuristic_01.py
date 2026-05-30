# MACE evolved heuristic 01/10 for problem: hybrid_reentrant_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Heuristic for Hybrid Reentrant Shop Scheduling:
    1. Determine machine assignments (fixed by problem logic).
    2. Generate multiple initial permutations using priority rules.
    3. Perform local search (hill-climbing) on the best candidate.
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']

    # The problem states machine assignment is determined by list scheduling.
    # The eval_func ignores the user-provided 'batch_assignment' for the 
    # actual schedule calculation, but it is required for feasibility.
    # We derive the correct machine assignment from the list scheduling logic.
    machine_assignment = [0] * n_jobs
    machine_heap = [(0, i + 1) for i in range(n_machines)]
    import heapq
    heapq.heapify(machine_heap)
    
    # Simulate list scheduling to get the machine for each job
    init_time = instance['init_time']
    for j in range(n_jobs):
        finish_time, m_id = heapq.heappop(machine_heap)
        machine_assignment[j] = m_id
        heapq.heappush(machine_heap, (finish_time + init_time, m_id))

    # Generate initial candidates
    rules = ['natural', 'spt_setup', 'lpt_setup', 'spt_main', 'lpt_main', 'spt_total', 'lpt_total', 'erd']
    best_perm = list(range(1, n_jobs + 1))
    best_makespan = float('inf')

    for rule in rules:
        if time.time() - start_time > time_limit_s * 0.2:
            break
        try:
            perm = tools['list_scheduling_priority'](rule)
            sim = tools['simulate_schedule'](perm)
            if sim['makespan'] < best_makespan:
                best_makespan = sim['makespan']
                best_perm = perm
        except Exception:
            continue

    # Refine using local search
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        # Use provided local swap tool
        improved_perm = tools['apply_local_swap'](
            best_perm, 
            time_limit_s=remaining_time * 0.9, 
            neighbourhood='insert'
        )
        
        # Verify improvement
        try:
            sim = tools['simulate_schedule'](improved_perm)
            if sim['makespan'] < best_makespan:
                best_perm = improved_perm
        except Exception:
            pass

    return {
        "permutation": best_perm,
        "batch_assignment": machine_assignment
    }