# MACE evolved heuristic 06/10 for problem: hybrid_reentrant_shop_scheduling
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Fixed Hybrid Reentrant Shop Scheduling solver.
    The primary issue was the potential for index errors in random permutations
    and ensuring robust permutation generation.
    """
    start_time = time.time()
    n_jobs = tools['n_jobs']()
    n_machines = tools['n_machines']()
    
    # 1. Deterministic Batch Assignment as per eval_func logic
    # The batch assignment for main processing is determined by the machine 
    # that performed the job's initialization.
    batch_assignment = [((j - 1) % n_machines) + 1 for j in range(1, n_jobs + 1)]
    
    # 2. Candidate Generation
    rules = ['natural', 'spt_setup', 'lpt_setup', 'spt_main', 'lpt_main', 'spt_total', 'lpt_total', 'erd']
    best_perm = None
    best_makespan = float('inf')
    
    for rule in rules:
        try:
            perm = tools['list_scheduling_priority'](rule=rule)
            sim = tools['simulate_schedule'](perm)
            if sim['makespan'] < best_makespan:
                best_makespan = sim['makespan']
                best_perm = list(perm)
        except Exception:
            continue
            
    # Fallback to random if rules failed or produced invalid results
    if best_perm is None or len(best_perm) != n_jobs:
        best_perm = list(range(1, n_jobs + 1))
        random.shuffle(best_perm)
        try:
            best_makespan = tools['simulate_schedule'](best_perm)['makespan']
        except Exception:
            # If even random fails, return a basic natural permutation
            best_perm = list(range(1, n_jobs + 1))

    # 3. Simulated Annealing refinement
    current_perm = list(best_perm)
    temp = 100.0
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit_s * 0.85:
        # Decide move type
        if n_jobs < 2:
            break
            
        if random.random() < 0.5:
            # Swap
            i, j = random.sample(range(n_jobs), 2)
            neighbor = list(current_perm)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Insert
            i, j = random.sample(range(n_jobs), 2)
            neighbor = list(current_perm)
            val = neighbor.pop(i)
            neighbor.insert(j, val)
            
        try:
            sim = tools['simulate_schedule'](neighbor)
            new_makespan = sim['makespan']
            
            # Acceptance criteria
            if new_makespan < best_makespan:
                best_makespan = new_makespan
                best_perm = list(neighbor)
                current_perm = list(neighbor)
            elif random.random() < math.exp((best_makespan - new_makespan) / (temp + 1e-9)):
                current_perm = list(neighbor)
            
            temp *= cooling_rate
        except Exception:
            continue
            
    return {
        "permutation": best_perm,
        "batch_assignment": batch_assignment
    }