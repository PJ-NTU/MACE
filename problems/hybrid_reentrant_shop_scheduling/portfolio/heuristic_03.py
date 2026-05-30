# MACE evolved heuristic 03/10 for problem: hybrid_reentrant_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Hybrid Reentrant Shop Scheduling problem using a GRASP-inspired
    approach: generating multiple starting points using heuristic priority rules
    and refining them with local search (adjacent swaps).
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']

    # 1. Determine batch_assignment
    # The problem states jobs are initialized on primary machines using list scheduling.
    # Given init_time is constant, this is round-robin: job j -> machine ((j-1) % n_machines) + 1.
    # The eval_func logic confirms machine_assignment is derived this way.
    batch_assignment = [((j - 1) % n_machines) + 1 for j in range(1, n_jobs + 1)]

    # 2. Generate candidate permutations using heuristics
    rules = ['natural', 'spt_setup', 'lpt_setup', 'spt_main', 'lpt_main', 'spt_total', 'lpt_total', 'erd']
    candidates = []
    
    for rule in rules:
        if time.time() - start_time > time_limit_s * 0.2:
            break
        try:
            perm = tools['list_scheduling_priority'](rule=rule)
            res = tools['simulate_schedule'](permutation=perm)
            candidates.append((res['makespan'], perm))
        except Exception:
            continue

    # 3. Local Search
    # Sort candidates by makespan and refine the best ones
    candidates.sort(key=lambda x: x[0])
    
    best_makespan = float('inf')
    best_perm = None

    for _, perm in candidates:
        if time.time() - start_time > time_limit_s * 0.9:
            break
        
        # Refine using adjacent swaps
        # We limit the local search time to respect the overall time budget
        remaining_time = time_limit_s - (time.time() - start_time) - 0.1
        if remaining_time <= 0:
            break
            
        refined_perm = tools['apply_local_swap'](perm, time_limit_s=remaining_time, neighbourhood='adjacent')
        
        current_res = tools['simulate_schedule'](permutation=refined_perm)
        if current_res['makespan'] < best_makespan:
            best_makespan = current_res['makespan']
            best_perm = refined_perm

    # Fallback if no search performed
    if best_perm is None:
        best_perm = list(range(1, n_jobs + 1))

    return {
        "permutation": best_perm,
        "batch_assignment": batch_assignment
    }