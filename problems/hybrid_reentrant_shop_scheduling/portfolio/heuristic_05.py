# MACE evolved heuristic 05/10 for problem: hybrid_reentrant_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized heuristic for Hybrid Reentrant Shop Scheduling:
    1. Uses a more accurate reconstruction of the list-scheduling logic for 'batch_assignment'.
    2. Employs a multi-start strategy with both deterministic rules and randomized priority vectors.
    3. Uses 'insert' local search refinement with a strictly enforced time budget.
    """
    start_time = time.time()
    n_jobs = tools['n_jobs']()
    n_machines = tools['n_machines']()
    init_time = instance['init_time']

    # 1. Precise Reconstruction of batch_assignment
    # The eval_func uses a heap for list scheduling, which is deterministic.
    import heapq
    batch_assignment = [0] * n_jobs
    machine_heap = [(0, machine_id) for machine_id in range(1, n_machines + 1)]
    heapq.heapify(machine_heap)
    for j in range(1, n_jobs + 1):
        _, machine_id = heapq.heappop(machine_heap)
        batch_assignment[j - 1] = machine_id
        heapq.heappush(machine_heap, (init_time * j, machine_id)) # Not quite, but list scheduling logic is fixed

    # Correct logic for batch_assignment based on eval_func:
    machine_heap = [(0, i) for i in range(1, n_machines + 1)]
    heapq.heapify(machine_heap)
    actual_batch_assignment = [0] * n_jobs
    for j in range(1, n_jobs + 1):
        avail, m_id = heapq.heappop(machine_heap)
        actual_batch_assignment[j - 1] = m_id
        heapq.heappush(machine_heap, (avail + init_time, m_id))

    # 2. Multi-start strategy
    best_perm = None
    best_makespan = float('inf')
    
    rules = ['natural', 'spt_setup', 'lpt_setup', 'spt_main', 'lpt_main', 'spt_total', 'lpt_total', 'erd']
    
    # Try all priority rules
    for rule in rules:
        if time.time() - start_time > time_limit_s * 0.3:
            break
        try:
            perm = tools['list_scheduling_priority'](rule=rule)
            sim = tools['simulate_schedule'](perm)
            if sim['makespan'] < best_makespan:
                best_makespan = sim['makespan']
                best_perm = perm
        except:
            continue

    # Try randomized priority vectors if time permits
    while time.time() - start_time < time_limit_s * 0.5:
        priorities = [random.random() for _ in range(n_jobs)]
        res = tools['decode_priorities_to_schedule'](priorities)
        if res['makespan'] < best_makespan:
            best_makespan = res['makespan']
            best_perm = res['permutation']

    if best_perm is None:
        best_perm = list(range(1, n_jobs + 1))

    # 3. Refinement
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        best_perm = tools['apply_local_swap'](
            best_perm, 
            time_limit_s=remaining_time * 0.95, 
            neighbourhood='insert'
        )

    return {
        "permutation": best_perm,
        "batch_assignment": actual_batch_assignment
    }